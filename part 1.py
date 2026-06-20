import math
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Device Setup
dev = qml.device("default.qubit", wires=4)

# 2. Parameterized Quantum Circuit Node
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# 3. Quantum Layer with Identity / Zero Initialization
class QuantumLinearLayer(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        # Shape rule for StronglyEntanglingLayers: (layers, qubits, 3)
        weight_shapes = {"weights": (num_layers, 4, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # --- INNOVATION STEP: Identity / Zero Initialization ---
        # We manually reach into the PyTorch parameter and fill it with zeros
        with torch.no_grad():
            self.qlayer.weights.zero_() 

    def forward(self, x):
        return self.qlayer(x)

# 4. Quantum Attention Engine
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
        context_vector = torch.bmm(attention_weights, V)
        return context_vector

# 5. Gradient Test and Optimization Loop
if __name__ == "__main__":
    model = QuantumSelfAttention(embed_dim=4, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # Input sequence
    seq_input = torch.tensor([
        [[0.5, -0.2, 0.1, 0.9], [0.1, 0.4, -0.8, 0.3], [0.0, 0.9, -0.1, -0.5]],
        [[0.2, 0.3, 0.5, -0.1], [-0.4, -0.2, 0.8, 0.1], [0.7, -0.7, 0.2, 0.0]]
    ], dtype=torch.float32)
    
    # Fake Target Output Sequence we want the model to learn to predict
    target = torch.randn(2, 3, 4)
    criterion = nn.MSELoss()

    print("\n" + "="*50)
    print("STARTING GRADIENT ENGINE TRACKING")
    print("="*50)

    for epoch in range(3):
        optimizer.zero_grad()
        
        # Forward Pass
        output = model(seq_input)
        loss = criterion(output, target)
        
        # Backward Pass
        loss.backward()
        
        # Extract gradient metrics for the Q-layer weights to verify life signals
        q_grads = model.q_layer.qlayer.qnode_weights['weights'].grad
        grad_mean = q_grads.mean().item()
        grad_std = q_grads.std().item()
        
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
        print(f" -> Q-Layer Weight Gradients Mean: {grad_mean:.6f}")
        print(f" -> Q-Layer Weight Gradients Std:  {grad_std:.6f}\n")
        
        optimizer.step()
        
    print("="*50)