import math
import pennylane as qml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("Generating Barren Plateau Comparison Plot...")

qubit_ranges = [4, 6, 8, 10]
random_stds = []
identity_stds = []

for num_qubits in qubit_ranges:
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    weight_shapes = {"weights": (2, num_qubits, 3)}
    
    # Random Init
    layer_rand = qml.qnn.TorchLayer(circuit, weight_shapes)
    dummy_in = torch.randn(1, num_qubits)
    loss_r = nn.MSELoss()(layer_rand(dummy_in), torch.randn(1, num_qubits))
    loss_r.backward()
    random_stds.append(layer_rand.qnode_weights['weights'].grad.std().item())

    # Identity Init
    layer_ident = qml.qnn.TorchLayer(circuit, weight_shapes)
    with torch.no_grad():
        layer_ident.weights.zero_()
    loss_i = nn.MSELoss()(layer_ident(dummy_in), torch.randn(1, num_qubits))
    loss_i.backward()
    identity_stds.append(layer_ident.qnode_weights['weights'].grad.std().item())

# --- PLOTTING ENGINE ---
plt.figure(figsize=(8, 5))
plt.plot(qubit_ranges, random_stds, label="Standard Random Init (Barren Plateau Vulnerable)", marker='o', color='red', linewidth=2)
plt.plot(qubit_ranges, identity_stds, label="Identity Block Init (Ours)", marker='s', color='green', linewidth=2)

plt.yscale("log") # Standard academic log-scale for variance decay
plt.xlabel("Number of Qubits (Model Scale)", fontsize=11)
plt.ylabel("Gradient Variance / Std (Log Scale)", fontsize=11)
plt.title("Mitigating Barren Plateaus in Quantum Attention Architectures", fontsize=12, fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10)

# Save the plot directly to your workspace folder
plt.savefig("barren_plateau_mitigation.png", dpi=300, bbox_inches='tight')
print("\nSuccess! Plot saved to your workspace as 'barren_plateau_mitigation.png'")