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
        """Compute a composite training capability score based on stay time, compute power, and transmission rate."""
        norm_stay_time = self.stay_time / 3600  # Normalize stay time (max 1 hour)
        norm_compute_power = self.compute_power / 1e9  # Normalize compute power (max 1 GFLOPS)
        norm_transmission_rate = self.transmission_rate / 100  # Normalize transmission rate (max 100 Mbps)
        return 0.4 * norm_stay_time + 0.4 * norm_compute_power + 0.2 * norm_transmission_rate


def initialize_vehicle_terminals(num_clients):
    """Initialize vehicle terminals with random attributes."""
    terminals = []
    for i in range(num_clients):
        speed = np.random.uniform(0, 120)  # 0-120 km/h
        compute_power = np.random.uniform(0.1, 1)  # 0.1-1 GFLOPs

        transmission_rate = np.random.uniform(10, 100)  # 10-100 Mbps
        stay_time = np.random.uniform(60, 3600)  # 1-60 minutes
        terminal = VehicleTerminal(i, speed, compute_power, transmission_rate, stay_time)
        terminals.append(terminal)
    return terminals


def dynamic_clustering(terminals, coverage_threshold=0.9, params_per_flop=0.05):
    """Perform dynamic clustering using K-means until total parameter coverage rate reaches threshold."""
    capabilities = np.array([t.training_capability for t in terminals]).reshape(-1, 1)
    k = 2
    clusters = []
    total_coverage = 0.0

    while total_coverage < coverage_threshold:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(capabilities)
        cluster_coverage = []
        cluster_terminals = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            cluster_terminals[label].append(terminals[i])

        for cluster in cluster_terminals:
            if not cluster:
                continue
            avg_compute_power = np.mean([t.compute_power for t in cluster])
            coverage = (avg_compute_power * params_per_flop)  # Normalize by 1B parameters

            cluster_coverage.append(coverage)

        total_coverage = sum(cluster_coverage)
        clusters = cluster_terminals
        if total_coverage < coverage_threshold:
            k += 1
            if k > len(terminals):
                break

    client_groups = {i: [t.id for t in cluster] for i, cluster in enumerate(clusters)}
    print(client_groups)
    print(f"Clustered into {len(clusters)} groups with total coverage rate: {total_coverage:.2%}")
    return client_groups


class FedCET:
    def __init__(self, model, sparsity, init_explore_fraction, end_round, update_frequency,
                 num_groups, num_clients, clients_per_round, local_epochs, device='cuda',
                 coverage_threshold=0.9, params_per_flop=100, client_groups=None):
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
        # Initialize vehicle terminals and perform clustering
        if client_groups is None:
            terminals = initialize_vehicle_terminals(num_clients)
            self.client_groups = dynamic_clustering(terminals, coverage_threshold, params_per_flop)
            self.num_groups = len(self.client_groups)  # Update num_groups based on clustering
        else:
            self.client_groups = client_groups
        self.group_masks = {}
        self.accuracy_history = []

    def _assign_clients_to_groups(self):
        client_groups = {}
        clients_per_group = self.num_clients // self.num_groups
        remaining = self.num_clients % self.num_groups
        client_ids = list(range(self.num_clients))
        random.shuffle(client_ids)
        start_idx = 0
        for g in range(self.num_groups):
            group_size = clients_per_group + (1 if g < remaining else 0)
            client_groups[g] = client_ids[start_idx:start_idx + group_size]
            start_idx += group_size
        return client_groups

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

    def _get_explore_fraction(self, round_num):
        if round_num >= self.end_round:
            return 0.0
        return self.init_explore_fraction * 0.5 * (1 + math.cos(round_num * math.pi / self.end_round))

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

    def _select_parameters(self, client_models, client_data_sizes, round_num):
        if round_num >= self.end_round:
            return
        sensitivity_scores = {}
        group_client_counts = {}
        for name, _ in self.global_model.named_parameters():
            if 'bias' not in name and 'bn' not in name:
                sensitivity_scores[name] = torch.zeros_like(self.global_model.state_dict()[name])
        group_models = {g: [] for g in range(self.num_groups)}
        group_data_sizes = {g: [] for g in range(self.num_groups)}
        for client_id, model in client_models.items():
            group_id = next(g for g, clients in self.client_groups.items() if client_id in clients)
            group_models[group_id].append(model)
            group_data_sizes[group_id].append(client_data_sizes[client_id])
        for g in range(self.num_groups):
            if not group_models[g]:
                continue
            group_model = copy.deepcopy(self.global_model)
            aggregated_dict = OrderedDict()
            total_data_size = sum(group_data_sizes[g])
            for name, param in group_model.named_parameters():
                aggregated_param = torch.zeros_like(param)
                for i, model in enumerate(group_models[g]):
                    weight = group_data_sizes[g][i] / total_data_size
                    aggregated_param += weight * model.state_dict()[name]
                aggregated_dict[name] = aggregated_param
            group_model.load_state_dict(aggregated_dict)
            for name, param in group_model.named_parameters():
                if 'bias' not in name and 'bn' not in name:
                    sensitivity_scores[name] += torch.abs(param) * self.group_masks[g][name]
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
            if round_num % self.update_frequency == 0 and round_num < self.end_round:
                self._explore_group_structures(round_num)
            selected_clients = random.sample(range(self.num_clients), self.clients_per_round)
            client_models = {}
            client_data_sizes = {}
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
                client_models[client_id] = client_model
                client_data_sizes[client_id] = data_size
            self._aggregate_models(client_models, client_data_sizes, round_num)
            if round_num % self.update_frequency == 0 and round_num < self.end_round:
                self._select_parameters(client_models, client_data_sizes, round_num)
            accuracy = self._evaluate(test_data)
            self.accuracy_history.append(accuracy)
            pbar.set_description(f"Round {round_num + 1}/{num_rounds}, Accuracy: {accuracy:.2%}")
            if (round_num + 1) % 50 == 0 or round_num == num_rounds - 1:
                print(f"Round {round_num + 1}/{num_rounds}, Test Accuracy: {accuracy:.2%}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = copy.deepcopy(self.global_model)
        print(f"Best Test Accuracy: {best_accuracy:.2%}")
        return best_model, best_accuracy, self.accuracy_history

    def _aggregate_models(self, client_models, client_data_sizes, round_num):
        """Aggregate client models according to the FedCET strategy."""
        if not client_models:
            return
        aggregated_dict = OrderedDict()
        total_data_size = sum(client_data_sizes.values())
        if round_num < self.end_round:
            for name, param in self.global_model.named_parameters():
                aggregated_dict[name] = torch.zeros_like(param)
            for client_id, model in client_models.items():
                weight = client_data_sizes[client_id] / total_data_size
                group_id = next(g for g, clients in self.client_groups.items() if client_id in clients)
                client_mask = self.group_masks[group_id]
                for name, param in model.named_parameters():
                    if name in self.global_mask:
                        public_mask = self.global_mask[name]
                        private_mask = client_mask[name] - public_mask
                        private_mask = torch.clamp(private_mask, 0, 1)
                        aggregated_dict[name] += weight * param.data * public_mask
            for g in range(self.num_groups):
                group_clients = [c for c in client_models if c in self.client_groups[g]]
                if not group_clients:
                    continue
                group_data_size = sum(client_data_sizes[c] for c in group_clients)
                for client_id in group_clients:
                    weight = client_data_sizes[client_id] / group_data_size
                    model = client_models[client_id]
                    for name, param in model.named_parameters():
                        if name in self.global_mask:
                            public_mask = self.global_mask[name]
                            private_mask = self.group_masks[g][name] - public_mask
                            private_mask = torch.clamp(private_mask, 0, 1)
                            aggregated_dict[name] += weight * param.data * private_mask
        else:
            for name, param in self.global_model.named_parameters():
                aggregated_dict[name] = torch.zeros_like(param)
                for client_id, model in client_models.items():
                    weight = client_data_sizes[client_id] / total_data_size
                    aggregated_dict[name] += weight * model.state_dict()[name]
        for name, param in self.global_model.named_parameters():
            if name in self.global_mask:
                if round_num < self.end_round:
                    mask = torch.zeros_like(param)
                    mask += self.global_mask[name]
                    for g in range(self.num_groups):
                        if any(c in client_models for c in self.client_groups[g]):
                            private_mask = self.group_masks[g][name] - self.global_mask[name]
                            private_mask = torch.clamp(private_mask, 0, 1)
                            mask += private_mask
                    mask = torch.clamp(mask, 0, 1)
                    param.data = aggregated_dict[name] * mask
                else:
                    param.data = aggregated_dict[name] * self.global_mask[name]
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
    fedcet = FedCET(
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
        params_per_flop=params_per_flop
    )

    def client_optimizer_fn(params):
        return optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)

    best_model, best_acc, accuracy_history = fedcet.train(
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
        plt.title('FedCET Accuracy per Round')
        plt.grid(True)
        plt.savefig('fedcet_accuracy.png')
        print("Accuracy plot saved as 'fedcet_accuracy.png'")
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
