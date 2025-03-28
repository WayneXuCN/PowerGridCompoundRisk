import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from scipy.sparse import csr_matrix
import numpy as np


class HighOrderSimplexConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(HighOrderSimplexConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, L1_tilde):
        # Apply Hodge Laplacian and linear transformation
        x = torch.matmul(L1_tilde, x)
        x = self.lin(x)
        return F.relu(x)


class HighOrderGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(HighOrderGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HighOrderSimplexConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(HighOrderSimplexConv(hidden_dim, hidden_dim))
        self.final_lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, L1_tilde):
        for conv in self.convs:
            x = conv(x, edge_index, L1_tilde)
        x = self.final_lin(x)
        return torch.sigmoid(x)


def train_gnn(adj_sparse, B1, L1_tilde, distances, initial_failure_labels):
    # Convert data to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Edge features: [distance, flood risk sum, heatwave risk sum]
    edge_features = np.hstack([distances, flood_risks, heatwave_risks])
    edge_features = torch.tensor(edge_features, dtype=torch.float).to(device)

    # Labels
    labels = torch.tensor(initial_failure_labels, dtype=torch.float).to(device)

    # Model
    model = HighOrderGNN(input_dim=edge_features.shape[1], hidden_dim=64, output_dim=1, num_layers=2).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(edge_features, None, L1_tilde)
        loss = criterion(predictions.squeeze(), labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


if __name__ == "__main__":
    # Load preprocessed data
    adj_sparse = sp.load_npz(PROCESSED_DATA_DIR / "adj_sparse.npz")
    B1 = np.load(PROCESSED_DATA_DIR / "B1.npy")
    L1_tilde = sp.load_npz(PROCESSED_DATA_DIR / "L1_tilde.npz")
    distances = np.load(PROCESSED_DATA_DIR / "distances.npy")

    # Define initial failure labels (example: southern nodes with lat < 45)
    initial_failure_labels = (latitudes < 45).astype(float)
    initial_failure_labels[:int(len(initial_failure_labels) * 0.1)] = 1.0

    # Train GNN
    model = train_gnn(adj_sparse, B1, L1_tilde, distances, initial_failure_labels)