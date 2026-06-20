import math
import pennylane as qml
import torch
import torch.nn as nn

print("\n" + "="*50)
print("RUNNING EXTREME SCALE QUANTUM GRADIENT BENCHMARK")
print("="*50 + "\n")

# We will test circuit scales from 4 qubits up to 10 qubits
qubit_ranges = [4, 6, 8, 10]

for num_qubits in qubit_ranges:
    # 1. Dynamically create a device for this scale
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def dynamic_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    # 2. Setup Random Model
    weight_shapes = {"weights": (2, num_qubits, 3)}
    
    random_layer = qml.qnn.TorchLayer(dynamic_circuit, weight_shapes)
    identity_layer = qml.qnn.TorchLayer(dynamic_circuit, weight_shapes)
    
    # Force Identity Initialization on the second layer
    with torch.no_grad():
        identity_layer.weights.zero_()

    # 3. Create dummy inputs matching the qubit scale
    # Shape: (batch_size=1, sequence_length=1, dimensions=num_qubits)
    dummy_input = torch.randn(1, num_qubits)
    target = torch.randn(1, num_qubits)
    criterion = nn.MSELoss()

    # ---- TEST RANDOM GRADIENTS ----
    out_rand = random_layer(dummy_input)
    loss_rand = criterion(out_rand, target)
    loss_rand.backward()
    grad_std_rand = random_layer.qnode_weights['weights'].grad.std().item()

    # ---- TEST IDENTITY GRADIENTS ----
    out_ident = identity_layer(dummy_input)
    loss_ident = criterion(out_ident, target)
    loss_ident.backward()
    grad_std_ident = identity_layer.qnode_weights['weights'].grad.std().item()

    print(f"Qubits: {num_qubits} | Dimensions: {num_qubits}")
    print(f"  -> Random Init Grad Std:   {grad_std_rand:.8f}")
    print(f"  -> Identity Init Grad Std: {grad_std_ident:.8f}")
    print("-" * 40)