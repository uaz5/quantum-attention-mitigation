import math
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Quantum Hardware Device Simulator
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QuantumLinearLayer(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        weight_shapes = {"weights": (num_layers, 4, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        # Identity Initialization to prevent Barren Plateaus
        with torch.no_grad():
            self.qlayer.weights.zero_() 

    def forward(self, x):
        return self.qlayer(x)

# 2. Quantum Attention Module
class QuantumSelfAttention(nn.Module):
    def __init__(self, embed_dim=4, num_layers=2):
        super().__init__()
        self.q_layer = QuantumLinearLayer(num_layers=num_layers)
        self.k_layer = QuantumLinearLayer(num_layers=num_layers)
        self.v_layer = QuantumLinearLayer(num_layers=num_layers)
        self.softmax = nn.Softmax(dim=-1)

    def _process_quantum(self, layer, x):
        B, L, D = x.shape
        flattened = x.view(B * L, D)
        q_features = layer(flattened)
        return q_features.view(B, L, D)

    def forward(self, x):
        B, L, D = x.shape
        Q = self._process_quantum(self.q_layer, x)
        K = self._process_quantum(self.k_layer, x)
        V = self._process_quantum(self.v_layer, x)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
        attention_weights = self.softmax(scores)
        return torch.bmm(attention_weights, V)

# 3. Execution & Training Engine
if __name__ == "__main__":
    print("\n" + "="*50)
    print("INITIALIZING QUANTUM MODEL TRAINING PIPELINE")
    print("="*50 + "\n")
    
    model = QuantumSelfAttention(embed_dim=4, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.MSELoss()

    # Fixed dataset: A simple sequence pattern
    # Shape: (batch_size=1, sequence_length=5, embedding_dim=4)
    X = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                       [0.5, 0.6, 0.7, 0.8],
                       [0.9, 0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6, 0.7],
                       [0.8, 0.9, 0.1, 0.2]]], dtype=torch.float32)
    
    # Target is a highly non-linear modification of the input (what the model must learn)
    Y = torch.sin(X) * 0.5

    # Training Loop
    epochs = 15
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass through the quantum attention module
        predictions = model(X)
        
        # Calculate Mean Squared Error Loss
        loss = criterion(predictions, Y)
        
        # Backpropagation through the quantum state simulator
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Current Loss: {loss.item():.6f}")

    print("\n" + "="*50)
    print("TRAINING COMPLETE. QUANTUM STATES OPTIMIZED.")
    print("="*50 + "\n")