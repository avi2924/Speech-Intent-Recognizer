import torch
import torch.nn as nn
import torch.optim as optim
import time

print("Step 1: Initializing CUDA...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Step 2: Creating a simple model...")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel().to(device)
print("Model created and moved to GPU")

print("Step 3: Creating loss function...")
criterion = nn.CrossEntropyLoss()
print("Loss function created")

print("Step 4: Creating optimizer...")
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Optimizer created")

print("Step 5: Creating dummy data...")
inputs = torch.randn(4, 100).to(device)
targets = torch.randint(0, 10, (4,)).to(device)
print("Data created and moved to GPU")

print("Step 6: Forward pass...")
start_time = time.time()
outputs = model(inputs)
print(f"Forward pass completed in {time.time() - start_time:.4f} seconds")

print("Step 7: Computing loss...")
start_time = time.time()
loss = criterion(outputs, targets)
print(f"Loss computed in {time.time() - start_time:.4f} seconds")

print("Step 8: Backward pass...")
start_time = time.time()
loss.backward()
print(f"Backward pass completed in {time.time() - start_time:.4f} seconds")

print("Step 9: Optimizer step...")
start_time = time.time()
optimizer.step()
print(f"Optimizer step completed in {time.time() - start_time:.4f} seconds")

print("All steps completed successfully!")