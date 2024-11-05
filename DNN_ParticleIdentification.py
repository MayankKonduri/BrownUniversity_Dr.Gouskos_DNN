import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import torch_geometric.transforms as T
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def create_dummy_graph(num_nodes=100, input_dim=3, k=8):
    x = torch.randn((num_nodes, input_dim))  # Features for each node
    edge_index = knn_graph(x, k=k)  # Create kNN graph
    y = torch.randint(0, 2, (num_nodes,))  # Binary classification labels for each node
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

data = create_dummy_graph()

# Graph Neural Network Model
class GraphDNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphDNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # First GCN Layer
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # Second GCN Layer
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Global mean pooling
        x = self.fc(x)  # Fully connected layer
        return F.log_softmax(x, dim=1)

def visualize_graph(data, k=8):
    edge_index = data.edge_index.cpu().numpy()
    pos = {i: [data.x[i, 0].item(), data.x[i, 1].item()] for i in range(data.num_nodes)}
    G = nx.Graph()
    
    for i in range(data.num_nodes):
        G.add_node(i)
    
    for start, end in edge_index.T:
        G.add_edge(int(start), int(end), edge_type='knn')
    
    plt.figure(figsize=(12, 8))
    knn_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == 'knn']
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='k')
    nx.draw_networkx_edges(G, pos, edgelist=knn_edges, edge_color='r', width=2)
    plt.axis('off')
    plt.show()

# Split the data into train and test sets
def train_test_split_data(data, train_size=0.8):
    indices = np.arange(data.num_nodes)
    train_indices, test_indices = train_test_split(indices, train_size=train_size)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

data = train_test_split_data(data)

# Training loop
def train_model(model, data, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

def test_model(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        accuracy = correct.sum().item() / data.test_mask.sum().item()
        print(f'Test Accuracy: {accuracy:.4f}')

# Initialize the model
input_dim = data.x.size(1)
hidden_dim = 16
output_dim = 2
model = GraphDNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Train the model
train_model(model, data, epochs=200, lr=0.01)

# Test the model
test_model(model, data)

# Visualize the graph
visualize_graph(data)
