import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import time

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

def log_step(step_name):
    """Log a step with timestamp"""
    logger.info(f"STEP: {step_name}")
    
# Seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

log_step("1. Initializing device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

log_step("2. Creating simple model")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

model = SimpleModel().to(device)
logger.info("Model created and moved to device")

log_step("3. Creating loss function")
criterion = nn.CrossEntropyLoss()
logger.info("Loss function created")

log_step("4. Creating optimizer (SGD)")
sgd_optimizer = optim.SGD(model.parameters(), lr=0.001)
logger.info("SGD optimizer created")

log_step("5. Creating optimizer (Adam)")
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)
logger.info("Adam optimizer created")

log_step("6. Creating dummy data")
inputs = torch.randn(4, 100).to(device)
targets = torch.randint(0, 10, (4,)).to(device)
logger.info("Data created and moved to device")

log_step("7. Forward pass")
outputs = model(inputs)
logger.info(f"Forward pass completed, output shape: {outputs.shape}")

log_step("8. Computing loss")
loss = criterion(outputs, targets)
logger.info(f"Loss computed: {loss.item()}")

log_step("9. SGD backward pass")
sgd_optimizer.zero_grad()
loss.backward()
logger.info("Backward pass completed")

log_step("10. SGD optimizer step")
sgd_optimizer.step()
logger.info("SGD optimizer step completed")

log_step("11. Adam backward pass with new forward pass")
# Make a new forward pass for Adam
outputs_adam = model(inputs)
loss_adam = criterion(outputs_adam, targets)
adam_optimizer.zero_grad()
loss_adam.backward()  # Use the new loss
logger.info("Adam backward pass completed")

log_step("12. Adam optimizer step")
adam_optimizer.step()
logger.info("Adam optimizer step completed")

log_step("13. Testing training loop")
# Simulate a mini training loop
for i in range(3):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    adam_optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    adam_optimizer.step()
    
    logger.info(f"Mini training iteration {i+1}/3 completed")

log_step("14. Testing iteration over DataLoader")
# Create a dummy dataset
dummy_data = [(torch.randn(100).to(device), i % 10) for i in range(10)]

# Create a DataLoader
from torch.utils.data import DataLoader, TensorDataset
logger.info("Creating DataLoader...")
tensor_x = torch.stack([x for x, _ in dummy_data])
tensor_y = torch.tensor([y for _, y in dummy_data])
dataset = TensorDataset(tensor_x, tensor_y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

logger.info("Iterating through DataLoader...")
for batch_idx, (data, target) in enumerate(loader):
    data, target = data.to(device), target.to(device)
    logger.info(f"Processed batch {batch_idx+1}/{len(loader)}")

log_step("ALL TESTS COMPLETED SUCCESSFULLY")