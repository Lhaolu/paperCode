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
import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from fedcet import FedCET, CNNModel, LeNet300_100, VGG11, ResNet18
from rfedcet import RFedCET


def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Federated Learning Framework ')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Experiment name for saving results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--algorithm', type=str, default='rfedcet', choices=['fedcet', 'rfedcet'],help='Algorithm to use (fedcet or rfedcet)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps', help='Device to run on (cuda/cpu)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing datasets')
    parser.add_argument('--non_iid', action='store_true', help='Use non-IID data distribution among clients')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='Alpha parameter for Dirichlet distribution for non-IID setting')
    parser.add_argument('--model', type=str, default='lenet', choices=['cnn', 'lenet', 'vgg11', 'resnet18'], help='Model architecture to use')
    parser.add_argument('--sparsity', type=float, default=0.9, help='Target sparsity level (0.0-1.0)')
    parser.add_argument('--explore_fraction', type=float, default=0.2, help='Initial fraction of parameters to explore (alpha)')
    parser.add_argument('--end_round', type=int, default=5, help='Round at which exploration ends (T_end)')
    parser.add_argument('--update_frequency', type=int, default=2, help='Frequency of structure updates (ΔT)')
    parser.add_argument('--num_groups', type=int, default=5, help='Number of groups for parallel exploration (Z)')
    parser.add_argument('--num_clients', type=int, default=100, help='Total number of clients (N)')
    parser.add_argument('--clients_per_round', type=int, default=20, help='Number of clients sampled per round (K)')
    parser.add_argument('--num_rounds', type=int, default=30, help='Total number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local training epochs (E)')
    parser.add_argument('--batch_size', type=int, default=32, help='Local batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size for testing')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--log_interval', type=int, default=1, help='How many rounds to wait before logging')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory for saving results')
    parser.add_argument('--coverage_threshold', type=float, default=0.9, help='Target parameter coverage rate for clustering')
    parser.add_argument('--params_per_flop', type=float, default=0.13, help='Parameters covered per FLOPS')
    # RFedCET parameters
    parser.add_argument('--loss_rate', type=float, default=0.1,help='Parameter loss rate for hypernetwork compensation (RFedCET)')
    parser.add_argument('--embed_dim', type=int, default=64, help='Dimension of embedding for hypernetwork (RFedCET)')
    parser.add_argument('--change_ratio_init', type=float, default=0.3,help='Initial ratio of groups to reassign (RFedCET)')
    parser.add_argument('--param_ratio_init', type=float, default=0.2,help='Initial ratio of parameters to transfer (RFedCET)')
    args = parser.parse_args()

    return args

def setup_experiment(args):
   
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.results_dir, f"{args.exp_name}_{args.algorithm}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    return exp_dir

def load_dataset(args):
    
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        num_classes = 10
        input_channels = 1
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 3
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
        input_channels = 3
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    return train_dataset, test_dataset, num_classes, input_channels

def distribute_data(train_dataset, args):
    
    num_clients = args.num_clients
    client_datasets = []
    if args.non_iid:
        if isinstance(train_dataset.targets, list):
            targets = torch.tensor(train_dataset.targets)
        else:
            targets = train_dataset.targets
        num_classes = len(torch.unique(targets))
        client_dict = {i: [] for i in range(num_clients)}
        class_priors = np.random.dirichlet(alpha=[args.dirichlet_alpha] * num_classes, size=num_clients)
        for c in range(num_classes):
            idx_c = torch.where(targets == c)[0]
            idx_c = idx_c[torch.randperm(len(idx_c))]
            proportions = class_priors[:, c] / np.sum(class_priors[:, c])
            cumulative_prop = np.cumsum(proportions)
            prev_idx = 0
            for i in range(num_clients):
                if i == num_clients - 1:
                    client_dict[i].extend(idx_c[prev_idx:].tolist())
                else:
                    idx = int(cumulative_prop[i] * len(idx_c))
                    client_dict[i].extend(idx_c[prev_idx:prev_idx+idx].tolist())
                    prev_idx += idx
        for i in range(num_clients):
            client_indices = client_dict[i]
            client_dataset = torch.utils.data.Subset(train_dataset, client_indices)
            client_loader = torch.utils.data.DataLoader(
                client_dataset, batch_size=args.batch_size, shuffle=True
            )
            client_datasets.append(client_loader)
    else:
        client_data_size = len(train_dataset) // num_clients
        indices = torch.randperm(len(train_dataset))
        for i in range(num_clients):
            start_idx = i * client_data_size
            end_idx = (i + 1) * client_data_size if i < num_clients - 1 else len(train_dataset)
            client_indices = indices[start_idx:end_idx]
            client_dataset = torch.utils.data.Subset(train_dataset, client_indices)
            client_loader = torch.utils.data.DataLoader(
                client_dataset, batch_size=args.batch_size, shuffle=True
            )
            client_datasets.append(client_loader)
    return client_datasets

def create_model(args, num_classes, input_channels):
    
    if args.model == 'cnn':
        model = CNNModel(num_classes=num_classes)
    elif args.model == 'lenet':
        model = LeNet300_100(num_classes=num_classes)
    elif args.model == 'vgg11':
        model = VGG11(num_classes=num_classes)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Model {args.model} not supported")
    return model.to(args.device)

def client_optimizer_factory(args):
    """Create a function that returns an optimizer for client training."""
    def client_optimizer_fn(params):
        return optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return client_optimizer_fn

def run_experiment(args):
    
    exp_dir = setup_experiment(args)
    print(f"Experiment directory: {exp_dir}")
    train_dataset, test_dataset, num_classes, input_channels = load_dataset(args)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False
    )
    client_datasets = distribute_data(train_dataset, args)
    print(f"Data distributed among {args.num_clients} clients")
    model = create_model(args, num_classes, input_channels)
    print(f"Created {args.model} model")

    if args.algorithm == 'fedcet':
        fed_algo = FedCET(
            model=model,
            sparsity=args.sparsity,
            init_explore_fraction=args.explore_fraction,
            end_round=args.end_round,
            update_frequency=args.update_frequency,
            num_groups=args.num_groups,
            num_clients=args.num_clients,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            device=args.device,
            coverage_threshold=args.coverage_threshold,
            params_per_flop=args.params_per_flop
        )
    elif args.algorithm == 'rfedcet':
        fed_algo = RFedCET(
            model=model,
            sparsity=args.sparsity,
            init_explore_fraction=args.explore_fraction,
            end_round=args.end_round,
            update_frequency=args.update_frequency,
            num_groups=args.num_groups,
            num_clients=args.num_clients,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            device=args.device,
            coverage_threshold=args.coverage_threshold,
            params_per_flop=args.params_per_flop,
            loss_rate=args.loss_rate,
            embed_dim=args.embed_dim,
            change_ratio_init=args.change_ratio_init,
            param_ratio_init=args.param_ratio_init
        )


    client_optimizer_fn = client_optimizer_factory(args)
    print(f"Starting {args.algorithm.upper()} training for {args.num_rounds} rounds...")
    best_model, best_acc, accuracy_history = fed_algo.train(
        train_data=client_datasets,
        test_data=test_loader,
        client_optimizer_fn=client_optimizer_fn,
        num_rounds=args.num_rounds
    )
    results = {
        'accuracy_history': accuracy_history,
        'best_accuracy': best_acc,
        'final_accuracy': accuracy_history[-1],
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(accuracy_history)), accuracy_history)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(f'{args.algorithm.upper()} Accuracy per Round (Sparsity: {args.sparsity}, Dataset: {args.dataset})')
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, 'accuracy_plot.png'))
    if args.save_model:
        torch.save(best_model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
        torch.save(fed_algo.global_model.state_dict(), os.path.join(exp_dir, 'final_model.pth'))
    accuracy_array = np.array(accuracy_history)
    top_indices = np.argsort(accuracy_array)[-10:][::-1]
    print("\nTop 10 accuracy values:")
    for i, idx in enumerate(top_indices):
        print(f"Rank {i+1}: Round {idx}, Accuracy: {accuracy_array[idx]:.4f}")
    final_sparsity = calculate_model_sparsity(fed_algo.global_model, fed_algo.global_mask)
    print(f"Final model sparsity: {final_sparsity:.2%}")
    print(f"Experiment completed. Results saved to {exp_dir}")
    return best_model, best_acc, accuracy_history, exp_dir

def calculate_model_sparsity(model, masks):

    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'bias' not in name and 'bn' not in name:
            total_params += param.numel()
            if name in masks:
                zero_params += param.numel() - torch.sum(masks[name]).item()
    return zero_params / total_params if total_params > 0 else 0

def compare_experiments(exp_dirs, labels):

    plt.figure(figsize=(12, 8))
    for exp_dir, label in zip(exp_dirs, labels):
        with open(os.path.join(exp_dir, 'results.json'), 'r') as f:
            results = json.load(f)
        with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        acc_history = results['accuracy_history']
        plt.plot(range(len(acc_history)), acc_history, label=label)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Experiments')
    plt.grid(True)
    plt.legend()
    comparison_dir = os.path.dirname(exp_dirs[0])
    plt.savefig(os.path.join(comparison_dir, 'comparison_plot.png'))
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    best_model, best_acc, accuracy_history, exp_dir = run_experiment(args)
