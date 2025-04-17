import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import math
from collections import OrderedDict
import random
from tqdm import tqdm
from sklearn.cluster import KMeans

class VehicleTerminal:
    def __init__(self, id, speed, compute_power, transmission_rate, stay_time):
        self.id = id
        self.speed = speed  # km/h
        self.compute_power = compute_power  # FLOPS
        self.transmission_rate = transmission_rate  # Mbps
        self.stay_time = stay_time  # seconds
        self.training_capability = self.compute_training_capability()

    def compute_training_capability(self):
        norm_stay_time = self.stay_time / 3600
        norm_compute_power = self.compute_power / 1e9
        norm_transmission_rate = self.transmission_rate / 100
        return 0.4 * norm_stay_time + 0.4 * norm_compute_power + 0.2 * norm_transmission_rate

def initialize_vehicle_terminals(num_clients):
    terminals = []
    for i in range(num_clients):
        speed = np.random.uniform(0, 120)
        compute_power = np.random.uniform(0.1, 1)
        transmission_rate = np.random.uniform(10, 100)
        stay_time = np.random.uniform(60, 3600)
        terminal = VehicleTerminal(i, speed, compute_power, transmission_rate, stay_time)
        terminals.append(terminal)
    return terminals

def dynamic_clustering(terminals, coverage_threshold=0.9, params_per_flop=0.05):
    capabilities = np.array([t.training_capability for t in terminals]).reshape(-1, 1)
    k = 2
    clusters = []
    head_nodes = []
    total_coverage = 0.0
    while total_coverage < coverage_threshold:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(capabilities)
        cluster_coverage = []
        cluster_terminals = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            cluster_terminals[label].append(terminals[i])
        cluster_coverage = []
        cluster_heads = []
        for cluster in cluster_terminals:
            if not cluster:
                continue
            avg_compute_power = np.mean([t.compute_power for t in cluster])
            head_terminal = max(cluster, key=lambda t: t.training_capability)
            cluster_heads.append(head_terminal)
            coverage = (avg_compute_power * params_per_flop)
            cluster_coverage.append(coverage)
        total_coverage = sum(cluster_coverage)
        clusters = cluster_terminals
        head_nodes = cluster_heads
        if total_coverage < coverage_threshold:
            k += 1
            if k > len(terminals):
                break
    client_groups = {i: [t.id for t in cluster] for i, cluster in enumerate(clusters)}
    head_node_ids = {i: head_terminal.id for i, head_terminal in enumerate(head_nodes)}
    print(client_groups)
    print(f"Clustered into {len(clusters)} groups with total coverage rate: {total_coverage:.2%}")
    return client_groups, clusters,head_node_ids

class HyperNetwork(nn.Module):
    def __init__(self, embed_dim, param_size):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, param_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RFedCET:
    def __init__(self, model, sparsity, init_explore_fraction, end_round, update_frequency,
                 num_groups, num_clients, clients_per_round, local_epochs, device='cuda',
                 coverage_threshold=0.9, params_per_flop=0.05, client_groups=None, loss_rate=0.1,
                 embed_dim=64, change_ratio_init=0.3, param_ratio_init=0.2):
        self.model = model
        self.sparsity = sparsity
        self.init_explore_fraction = init_explore_fraction
        self.end_round = end_round
        self.update_frequency = update_frequency
        self.num_groups = num_groups
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.local_epochs = local_epochs
        self.device = device
        self.coverage_threshold = coverage_threshold
        self.params_per_flop = params_per_flop
        self.loss_rate = loss_rate
        self.embed_dim = embed_dim
        self.change_ratio_init = change_ratio_init
        self.param_ratio_init = param_ratio_init
        self.global_model = copy.deepcopy(model).to(device)
        self.prunable_params = []
        self.param_name_to_idx = {}
        idx = 0
        for name, param in self.global_model.named_parameters():
            if 'bias' not in name and 'bn' not in name:
                self.prunable_params.append(param)
                self.param_name_to_idx[name] = idx
                idx += 1
        self.global_mask = None
        if client_groups is None:
            self.terminals = initialize_vehicle_terminals(num_clients)
            self.client_groups, self.clusters, self.head_node_ids = dynamic_clustering(
                self.terminals, coverage_threshold, params_per_flop)
            self.num_groups = len(self.client_groups)
        else:
            self.client_groups = client_groups
            self.terminals = initialize_vehicle_terminals(num_clients)
            self.clusters = [[self.terminals[i] for i in group] for group in client_groups.values()]
            self.head_node_ids = {i: max(cluster, key=lambda t: t.training_capability).id
                                  for i, cluster in enumerate(self.clusters)}

        self.group_masks = {}
        self.accuracy_history = []
        self.hyper_networks = {}
        for name, param in self.global_model.named_parameters():
            if 'bias' not in name and 'bn' not in name:
                self.hyper_networks[name] = HyperNetwork(embed_dim, param.numel()).to(device)
        self.hyper_optimizer = optim.Adam(
            [p for hn in self.hyper_networks.values() for p in hn.parameters()], lr=0.001
        )
        self.ucb_scores = {g: {'accuracy': [], 'latency': [], 'counts': 0, 'ucb': 0.0} for g in range(self.num_groups)}
        self.model_size = self._calculate_model_size()

    def _calculate_model_size(self):
        total_params = sum(p.numel() for name, p in self.global_model.named_parameters() if 'bias' not in name and 'bn' not in name)
        return total_params * 4 / 1e6  # Size in MB (assuming 4 bytes per parameter)

    def _compute_transmission_latency(self, group_id):
        cluster = self.clusters[group_id]
        avg_transmission_rate = np.mean([t.transmission_rate for t in cluster])  # Mbps
        latency = self.model_size / avg_transmission_rate  # seconds
        return latency

    def _get_explore_fraction(self, round_num):
        if round_num >= self.end_round:
            return 0.0
        return self.init_explore_fraction * 0.5 * (1 + math.cos(round_num * math.pi / self.end_round))

    def _get_change_ratio(self, round_num):
        if round_num >= self.end_round:
            return 0.0
        return self.change_ratio_init * 0.5 * (1 + math.cos(round_num * math.pi / self.end_round))

    def _get_param_ratio(self, round_num):
        if round_num >= self.end_round:
            return 0.0
        return self.param_ratio_init * 0.5 * (1 + math.cos(round_num * math.pi / self.end_round))

    def initialize_sparse_model(self, data_batch):
        print("Initializing sparse structure...")
        current_explore_fraction = self._get_explore_fraction(0)
        initial_keep_ratio = 1 - self.sparsity - (1 - self.sparsity) * current_explore_fraction
        self.global_mask = {}
        for name, param in self.global_model.named_parameters():
            if 'bias' not in name and 'bn' not in name:
                rand_mask = torch.rand_like(param)
                threshold = torch.quantile(rand_mask.flatten(), 1 - initial_keep_ratio)
                self.global_mask[name] = (rand_mask >= threshold).float()
            else:
                self.global_mask[name] = torch.ones_like(param)
        print(f"Initialized with {initial_keep_ratio:.2%} of parameters")

    def _explore_group_structures(self, round_num):
        explore_fraction = self._get_explore_fraction(round_num)
        if explore_fraction == 0:
            for g in range(self.num_groups):
                self.group_masks[g] = copy.deepcopy(self.global_mask)
            return
        for g in range(self.num_groups):
            group_mask = copy.deepcopy(self.global_mask)
            for name, mask in self.global_mask.items():
                if 'bias' not in name and 'bn' not in name:
                    non_zero_params = mask > 0
                    n_params = torch.sum(mask).item()
                    n_total = mask.numel()
                    n_explore = int(n_total * explore_fraction)
                    explore_mask = torch.zeros_like(mask)
                    zero_indices = torch.nonzero(mask == 0).squeeze()
                    if len(zero_indices.shape) == 0 and zero_indices.numel() > 0:
                        zero_indices = zero_indices.unsqueeze(0)
                    if zero_indices.numel() > 0:
                        perm = torch.randperm(zero_indices.size(0))
                        n_to_explore = min(n_explore, zero_indices.size(0))
                        selected_indices = zero_indices[perm[:n_to_explore]]
                        if n_to_explore == 1:
                            explore_mask.view(-1)[selected_indices] = 1.0
                        else:
                            explore_mask.view(-1)[selected_indices] = 1.0
                    group_mask[name] = (mask + explore_mask).clamp(0, 1)
            self.group_masks[g] = group_mask

    def _compute_ucb_scores(self, round_num):
        for g in range(self.num_groups):
            if self.ucb_scores[g]['counts'] == 0:
                self.ucb_scores[g]['ucb'] = float('inf')
            else:
                avg_accuracy = np.mean(self.ucb_scores[g]['accuracy']) if self.ucb_scores[g]['accuracy'] else 0.0
                latency = self._compute_transmission_latency(g)
                norm_latency = 1.0 / (1.0 + latency)  # Normalize latency to [0,1]
                reward = 0.7 * avg_accuracy + 0.3 * norm_latency
                exploration_term = math.sqrt(2 * math.log(round_num + 1) / self.ucb_scores[g]['counts'])
                self.ucb_scores[g]['ucb'] = reward + exploration_term

    def _reassign_groups(self, round_num):
        change_ratio = self._get_change_ratio(round_num)
        param_ratio = self._get_param_ratio(round_num)
        if change_ratio == 0 or param_ratio == 0:
            return
        sorted_groups = sorted(range(self.num_groups), key=lambda g: self.ucb_scores[g]['ucb'], reverse=True)
        high_perf_groups = sorted_groups[:int(self.num_groups * change_ratio)]
        low_perf_groups = sorted_groups[-int(self.num_groups * change_ratio):]
        for low_g in low_perf_groups:
            for name, mask in self.group_masks[low_g].items():
                if 'bias' not in name and 'bn' not in name:
                    non_zero_indices = torch.nonzero(mask).squeeze()
                    if non_zero_indices.numel() == 0:
                        continue
                    n_transfer = int(non_zero_indices.size(0) * param_ratio)
                    perm = torch.randperm(non_zero_indices.size(0))
                    transfer_indices = non_zero_indices[perm[:n_transfer]]
                    for high_g in high_perf_groups:
                        self.group_masks[high_g][name].view(-1)[transfer_indices] = 1.0
                    self.group_masks[low_g][name].view(-1)[transfer_indices] = 0.0

    def _select_parameters(self, cluster_head_models, cluster_data_sizes, round_num):
        if round_num >= self.end_round:
            return
        sensitivity_scores = {}
        for name, _ in self.global_model.named_parameters():
            if 'bias' not in name and 'bn' not in name:
                sensitivity_scores[name] = torch.zeros_like(self.global_model.state_dict()[name])
        for group_id, model in cluster_head_models.items():
            for name, param in model.named_parameters():
                if 'bias' not in name and 'bn' not in name:
                    sensitivity_scores[name] += torch.abs(param) * self.group_masks[group_id][name]
        explore_fraction = self._get_explore_fraction(round_num)
        keep_ratio = 1 - self.sparsity - (1 - self.sparsity) * explore_fraction
        all_scores = []
        for name, score in sensitivity_scores.items():
            all_scores.append(score.flatten())
        all_scores = torch.cat(all_scores)
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        if num_params_to_keep < 1:
            num_params_to_keep = 1
        threshold = torch.sort(all_scores)[0][len(all_scores) - num_params_to_keep]
        for name, score in sensitivity_scores.items():
            self.global_mask[name] = (score >= threshold).float()
        print(f"Updated global mask with {keep_ratio:.2%} of parameters at round {round_num}")

    def _generate_cluster_embedding(self, cluster_data_loaders):
        embedding = torch.zeros(self.embed_dim, device=self.device)
        count = 0
        for loader in cluster_data_loaders:
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                embedding += inputs.mean(dim=[0, 2, 3]).sum().item()
                count += 1
        return embedding / count if count > 0 else embedding

    def _compensate_parameters(self, cluster_head_models, cluster_data_loaders):
        compensated_models = {}
        # Calculate number of clusters to lose updates (ceil(num_groups * loss_rate))
        num_lost_clusters = math.ceil(self.num_groups * self.loss_rate)
        # Randomly select clusters to lose updates
        lost_group_ids = random.sample(list(cluster_head_models.keys()),
                                       min(num_lost_clusters, len(cluster_head_models)))
        print(
            f"Round: {self.current_round}, Total clusters: {len(cluster_head_models)}, Lost clusters: {len(lost_group_ids)} ({lost_group_ids})")

        for group_id, model in cluster_head_models.items():
            compensated_model = copy.deepcopy(model)
            if group_id in lost_group_ids:
                # Perform compensation for lost clusters
                embedding = self._generate_cluster_embedding(cluster_data_loaders[group_id])
                for name, param in compensated_model.named_parameters():
                    if 'bias' not in name and 'bn' not in name:
                        generated_param = self.hyper_networks[name](embedding).view(param.shape)
                        # Compute reconstruction loss for hypernetwork training
                        loss = F.mse_loss(generated_param, param.data)
                        self.hyper_optimizer.zero_grad()
                        loss.backward()
                        self.hyper_optimizer.step()
                        param.data = generated_param
            compensated_models[group_id] = compensated_model
        print(f"Compensated {len(lost_group_ids)} clusters")
        return compensated_models




    def train(self, train_data, test_data, client_optimizer_fn, num_rounds):
        sample_batch = next(iter(train_data[0]))
        self.initialize_sparse_model(sample_batch)
        self.accuracy_history = []
        best_accuracy = 0.0
        client_models = {}
        pbar = tqdm(range(num_rounds), desc="Training rounds")
        initial_accuracy = self._evaluate(test_data)
        self.accuracy_history.append(initial_accuracy)
        print(f"Initial Test Accuracy: {initial_accuracy:.2%}")
        for round_num in pbar:
            self.current_round = round_num
            if round_num % self.update_frequency == 0 and round_num < self.end_round:
                self._explore_group_structures(round_num)
                self._compute_ucb_scores(round_num)
                self._reassign_groups(round_num)
            # Select clients, ensuring at least one client per group if possible
            selected_groups = random.sample(range(self.num_groups), min(self.num_groups, self.clients_per_round))
            selected_clients = []
            for group_id in selected_groups:
                group_clients = self.client_groups[group_id]
                if group_clients:
                    selected_clients.append(self.head_node_ids[group_id])
            if len(selected_clients) < self.clients_per_round:
                remaining_clients = [i for i in range(self.num_clients) if i not in selected_clients]
                additional_clients = random.sample(remaining_clients,
                                                   min(self.clients_per_round - len(selected_clients),
                                                       len(remaining_clients)))
                selected_clients.extend(additional_clients)
            cluster_head_models = {}
            cluster_data_sizes = {}
            cluster_data_loaders = {}
            for client_id in selected_clients:
                group_id = next(g for g, clients in self.client_groups.items() if client_id in clients)
                client_mask = self.group_masks[group_id] if round_num < self.end_round else self.global_mask
                client_model = copy.deepcopy(self.global_model)
                for name, param in client_model.named_parameters():
                    if name in client_mask:
                        param.data = param.data * client_mask[name]
                client_optimizer = client_optimizer_fn(client_model.parameters())
                client_model.train()
                data_size = 0
                for epoch in range(self.local_epochs):
                    for batch_idx, (inputs, targets) in enumerate(train_data[client_id]):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        data_size += len(targets)
                        client_optimizer.zero_grad()
                        outputs = client_model(inputs)
                        loss = F.cross_entropy(outputs, targets)
                        loss.backward()
                        for name, param in client_model.named_parameters():
                            if name in client_mask:
                                param.grad.data = param.grad.data * client_mask[name]
                        client_optimizer.step()
                # Only head nodes contribute to cluster updates
                if client_id == self.head_node_ids.get(group_id):
                    cluster_head_models[group_id] = client_model
                    cluster_data_sizes[group_id] = data_size
                    cluster_data_loaders[group_id] = [train_data[c] for c in self.client_groups[group_id]]
            # Compensate parameters for cluster head models
            cluster_head_models = self._compensate_parameters(cluster_head_models, cluster_data_loaders)
            self._aggregate_models(cluster_head_models, cluster_data_sizes, round_num)
            if round_num % self.update_frequency == 0 and round_num < self.end_round:
                self._select_parameters(cluster_head_models, cluster_data_sizes, round_num)
            accuracy = self._evaluate(test_data)
            self.accuracy_history.append(accuracy)
            for group_id in cluster_head_models:
                self.ucb_scores[group_id]['accuracy'].append(accuracy)
                self.ucb_scores[group_id]['counts'] += 1
            pbar.set_description(f"Round {round_num + 1}/{num_rounds}, Accuracy: {accuracy:.2%}")
            if (round_num + 1) % 50 == 0 or round_num == num_rounds - 1:
                print(f"Round {round_num + 1}/{num_rounds}, Test Accuracy: {accuracy:.2%}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = copy.deepcopy(self.global_model)
        print(f"Best Test Accuracy: {best_accuracy:.2%}")
        return best_model, best_accuracy, self.accuracy_history

    def _aggregate_models(self, cluster_head_models, cluster_data_sizes, round_num):
        if not cluster_head_models:
            return
        aggregated_dict = OrderedDict()
        total_data_size = sum(cluster_data_sizes.values())
        if round_num < self.end_round:
            for name, param in self.global_model.named_parameters():
                aggregated_dict[name] = torch.zeros_like(param)
            for group_id, model in cluster_head_models.items():
                weight = cluster_data_sizes[group_id] / total_data_size
                for name, param in model.named_parameters():
                    if name in self.global_mask:
                        public_mask = self.global_mask[name]
                        private_mask = self.group_masks[group_id][name] - public_mask
                        private_mask = torch.clamp(private_mask, 0, 1)
                        aggregated_dict[name] += weight * param.data * (public_mask + private_mask)
        else:
            for name, param in self.global_model.named_parameters():
                aggregated_dict[name] = torch.zeros_like(param)
                for group_id, model in cluster_head_models.items():
                    weight = cluster_data_sizes[group_id] / total_data_size
                    aggregated_dict[name] += weight * model.state_dict()[name]
        for name, param in self.global_model.named_parameters():
            if name in self.global_mask:
                mask = self.global_mask[name]
                if round_num < self.end_round:
                    for group_id in cluster_head_models:
                        private_mask = self.group_masks[group_id][name] - self.global_mask[name]
                        private_mask = torch.clamp(private_mask, 0, 1)
                        mask += private_mask
                    mask = torch.clamp(mask, 0, 1)
                param.data = aggregated_dict[name] * mask
            else:
                param.data = aggregated_dict[name]

    def _evaluate(self, test_data):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

class LeNet300_100(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet300_100, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=100):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mnist_example():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)
    num_clients = 100
    client_data_size = len(mnist_train) // num_clients
    client_datasets = []
    indices = torch.randperm(len(mnist_train))
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = (i + 1) * client_data_size
        client_indices = indices[start_idx:end_idx]
        client_dataset = torch.utils.data.Subset(mnist_train, client_indices)
        client_loader = torch.utils.data.DataLoader(client_dataset, batch_size=32, shuffle=True)
        client_datasets.append(client_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet300_100().to(device)
    sparsity = 0.9
    init_explore_fraction = 0.2
    end_round = 5
    update_frequency = 2
    num_groups = 5
    clients_per_round = 20
    local_epochs = 5
    coverage_threshold = 0.9
    params_per_flop = 100
    loss_rate = 0.1
    embed_dim = 64
    rfedcet = RFedCET(
        model=model,
        sparsity=sparsity,
        init_explore_fraction=init_explore_fraction,
        end_round=end_round,
        update_frequency=update_frequency,
        num_groups=num_groups,
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        device=device,
        coverage_threshold=coverage_threshold,
        params_per_flop=params_per_flop,
        loss_rate=loss_rate,
        embed_dim=embed_dim
    )

    def client_optimizer_fn(params):
        return optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)

    best_model, best_acc, accuracy_history = rfedcet.train(
        train_data=client_datasets,
        test_data=test_loader,
        client_optimizer_fn=client_optimizer_fn,
        num_rounds=1000
    )
    print(f"Training completed. Best accuracy: {best_acc:.4f}")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(accuracy_history)), accuracy_history)
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('RFedCET Accuracy per Round')
        plt.grid(True)
        plt.savefig('rfedcet_accuracy.png')
        print("Accuracy plot saved as 'rfedcet_accuracy.png'")
    except ImportError:
        print("Could not generate plot: matplotlib not installed")
    return best_model, best_acc, accuracy_history

if __name__ == "__main__":
    best_model, best_acc, accuracy_history = mnist_example()
    accuracy_array = np.array(accuracy_history)
    top_indices = np.argsort(accuracy_array)[-10:][::-1]
    print("\nTop 10 accuracy values:")
    for i, idx in enumerate(top_indices):
        print(f"Rank {i + 1}: Round {idx}, Accuracy: {accuracy_array[idx]:.4f}")