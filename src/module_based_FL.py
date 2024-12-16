import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from threading import Thread
import random
import numpy as np
from tqdm import tqdm


# Define the base model
class BaseModel(nn.Module):
    def __init__(self, filter1=32, filter2=64, fc_size=256):
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(1, filter1, kernel_size=3, stride=1, padding=1)  # Adjusted for MNIST (1 channel input)
        self.conv2 = nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Reduce output size to (1x1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(filter2, 10)  # Output size for MNIST (10 classes)

        self.softmax = nn.Softmax(dim=1)  # Softmax for multi-class classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))

        x = torch.relu(self.conv2(x))
        x = self.gap(x)  # Apply 1x1 GAP

        x = self.flatten(x)

        x = self.fc(x)
        x = self.softmax(x)
        return x


# Split the layers by filters
def split_filters(layer, parts, in_c):
    """Split a convolutional layer into `parts` sub-layers."""
    split_size = layer.out_channels // parts
    sub_layers = []
    for i in range(parts):
        sub_layer = nn.Conv2d(
            in_channels=in_c,
            out_channels=split_size,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding
        )
        sub_layers.append(sub_layer)
    return sub_layers


# Generate all permutations of submodels
def generate_submodels():
    """Generate all permutations of submodels with consistent GAP and softmax output."""
    parts = 2  # Split each layer into 2 parts

    # Initialize base layers
    conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Adjusted for MNIST (1 channel input)
    conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

    # Split layers into parts
    in_ch = conv1.in_channels
    conv1_parts = split_filters(conv1, parts, in_ch)
    in_ch = conv1.out_channels // parts
    conv2_parts = split_filters(conv2, parts, in_ch)

    # Define GAP and FC layer
    fc_input_size = 64  # GAP reduces spatial dimensions to 1x1

    # Generate all combinations of submodels
    submodels = []

    for c1_part, c2_part in itertools.product(conv1_parts, conv2_parts):
        submodel = nn.Sequential(
            c1_part,
            nn.ReLU(),
            nn.Conv2d(c1_part.out_channels, c2_part.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Apply 1x1 GAP
            nn.Flatten(),
            nn.Linear(c2_part.out_channels, 10),  # Output layer with 10 values for MNIST classes
            nn.Softmax(dim=1)  # Apply softmax activation
        )
        submodels.append(submodel)

    return submodels


def aggregate_models(models, weights, client_assignments):
    """Aggregate the weights of multiple submodels using weighted average and specific concatenation."""
    aggregated_model = BaseModel()

    with torch.no_grad():
        # Aggregate conv1 weights: Average submodels 0 and 1, then concatenate with submodel 2
        submodel_0_indices = [i for i, assign in enumerate(client_assignments) if assign == 0]
        submodel_1_indices = [i for i, assign in enumerate(client_assignments) if assign == 1]
        submodel_2_indices = [i for i, assign in enumerate(client_assignments) if assign == 2]

        # Average and concatenate conv1 weights
        avg_conv1_01_weight = sum(models[i][0].weight.data for i in submodel_0_indices + submodel_1_indices) / (
                    len(submodel_0_indices) + len(submodel_1_indices))
        avg_conv1_01_bias = sum(models[i][0].bias.data for i in submodel_0_indices + submodel_1_indices) / (
                    len(submodel_0_indices) + len(submodel_1_indices))
        concat_conv1_weight = torch.cat([avg_conv1_01_weight, models[submodel_2_indices[0]][0].weight.data], dim=0)
        concat_conv1_bias = torch.cat([avg_conv1_01_bias, models[submodel_2_indices[0]][0].bias.data], dim=0)
        aggregated_model.conv1.weight.data = concat_conv1_weight
        aggregated_model.conv1.bias.data = concat_conv1_bias

        # Aggregate conv2 weights: Average submodels 0 and 1, then concatenate their input channels to match 32 channels
        avg_conv2_01_weight = sum(models[i][2].weight.data for i in submodel_0_indices + submodel_1_indices) / (
                    len(submodel_0_indices) + len(submodel_1_indices))
        avg_conv2_01_bias = sum(models[i][2].bias.data for i in submodel_0_indices + submodel_1_indices) / (
                    len(submodel_0_indices) + len(submodel_1_indices))
        # Ensure weights have 32 input channels by concatenating input slices
        concat_conv2_weight = torch.cat(
            [avg_conv2_01_weight[:, :, :, :], models[submodel_2_indices[0]][2].weight.data[:, :, :, :]],
            dim=1)  # Concatenate along input channel dimension
        concat_conv2_bias = torch.cat([avg_conv2_01_bias, models[submodel_2_indices[0]][2].bias.data], dim=0)

        concat_conv2_weight = torch.cat([concat_conv2_weight, concat_conv2_weight],
                                        dim=0)  # Duplicate averaged weights for 64 output channels

        aggregated_model.conv2.weight.data = concat_conv2_weight
        aggregated_model.conv2.bias.data = concat_conv2_bias
        avg_fc_01_weight = sum(models[i][-2].weight.data for i in submodel_0_indices + submodel_1_indices) / (
                    len(submodel_0_indices) + len(submodel_1_indices))

        concat_fc_weight = torch.cat([avg_fc_01_weight, models[submodel_2_indices[0]][-2].weight.data], dim=1)

        aggregated_model.fc.weight.data = concat_fc_weight
        aggregated_model.fc.bias.data = sum(
            w * model[-2].bias.data for model, w in zip(models, weights)
        ) / sum(weights)

    return aggregated_model


# Train a single submodel on MNIST
def train_submodel(model, train_loader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        print(f"local epoch {epoch}")
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            x = images
            output = model(images)

            loss = criterion(output, labels)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"accuracy: {accuracy}")
        print(f"avg_loss: {avg_loss}")

    return model


# Train submodels on MNIST using threads and shuffle assignments
def train_on_mnist(submodels, client_assignments, client_data_weights, epochs, dataset_size):
    # MNIST data loader (slice dataset to given size)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    subset_indices = list(range(dataset_size))  # Select a subset of the dataset
    train_dataset = Subset(full_dataset, subset_indices)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Shuffle submodel assignments
        random.shuffle(client_assignments)
        print(f"Client assignments: {client_assignments}")

        trained_submodels = [None] * len(client_assignments)

        def train_client(client, submodel_idx):
            model = submodels[submodel_idx]
            trained_submodels[client] = (train_submodel(model, train_loader, epochs=10), client_data_weights[client])

        threads = []
        for client, submodel_idx in enumerate(client_assignments):
            train_client(client, submodel_idx)

        # Extract models and weights for aggregation
        models, weights = zip(*trained_submodels)

        # Aggregate submodels
        aggregated_model = aggregate_models(models, weights, client_assignments)
        print("Aggregated model for epoch ready.")

        # Redistribute the aggregated model to submodels
        redistribute_weights(submodels, aggregated_model)

    return aggregated_model


def redistribute_weights(submodels, aggregated_model):
    for submodel in submodels:
        with torch.no_grad():
            # Assign weights and biases for the first conv layer
            submodel[0].weight.data = aggregated_model.conv1.weight.data[:submodel[0].out_channels]
            submodel[0].bias.data = aggregated_model.conv1.bias.data[:submodel[0].out_channels]

            # Assign weights and biases for the second conv layer
            submodel[2].weight.data = aggregated_model.conv2.weight.data[:submodel[2].out_channels,
                                      :submodel[2].in_channels]
            submodel[2].bias.data = aggregated_model.conv2.bias.data[:submodel[2].out_channels]

            # Slice weights for the fully connected layer
            fc_input_size = submodel[-2].in_features
            submodel[-2].weight.data = aggregated_model.fc.weight.data[:, :fc_input_size]
            submodel[-2].bias.data = aggregated_model.fc.bias.data


def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return accuracy, avg_loss


if __name__ == "__main__":
    # Generate all submodels
    submodels = generate_submodels()

    # Random client assignments (example: 5 clients, submodel indices)
    client_assignments = [0, 1, 0, 1, 2]  # Clients 0 and 2 share submodel 0, etc.
    client_data_weights = [1.0] * 5  # Example weights for client contributions

    # Train submodels on a sliced MNIST dataset (1000 samples)
    final_aggregated_model = train_on_mnist(submodels, client_assignments, client_data_weights, epochs=5,
                                            dataset_size=1000)
    print("Training complete.")

    # Evaluate the final aggregated model on the test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    accuracy, avg_loss = evaluate_model(final_aggregated_model, test_loader)
    print(f"Final Model Accuracy: {accuracy:.2f}%")
    print(f"Final Model Loss: {avg_loss:.4f}")
