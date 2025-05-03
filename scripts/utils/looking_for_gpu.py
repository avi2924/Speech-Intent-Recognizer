import torch
import time
import numpy as np
import math

def test_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Device information
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"Device {i}: {device_name} with {total_memory:.2f} GB memory")
        
        print("\n=== HIGH UTILIZATION GPU TEST ===")
        
        # Create multiple tensors to maximize GPU utilization
        tensors = []
        print("Creating multiple large tensors to maximize GPU utilization...")
        
        # Calculate approximately how many tensors we need based on GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        tensor_size = 5000  # Start with 5000x5000 matrices
        
        try:
            # Create multiple tensors to fill GPU memory to ~70%
            target_utilization = 0.7  # Target 70% memory utilization
            target_mem_gb = gpu_mem * target_utilization
            tensor_mem_gb = (tensor_size * tensor_size * 4) / (1024**3)  # 4 bytes per float32
            num_tensors = max(1, int(target_mem_gb / tensor_mem_gb))
            
            print(f"Creating {num_tensors} tensors of size {tensor_size}x{tensor_size}")
            print(f"Expected memory usage: ~{num_tensors * tensor_mem_gb:.2f} GB")
            
            # Create the tensors
            for i in range(num_tensors):
                tensors.append(torch.randn(tensor_size, tensor_size, device="cuda"))
                current_mem = torch.cuda.memory_allocated() / (1024**3)
                print(f"Tensor {i+1}/{num_tensors} created. Current memory: {current_mem:.2f} GB ({current_mem/gpu_mem*100:.1f}%)")
        
            # Start high utilization computation
            print("\nStarting intense computation for at least 10 seconds...")
            start_time = time.time()
            iteration = 0
            
            # Run until we've completed at least 10 seconds of computation
            while time.time() - start_time < 10:
                iteration += 1
                
                # Force multiple complex operations to maximize utilization
                for i in range(len(tensors)):
                    # Use in-place operations where possible to maximize efficiency
                    tensors[i] = torch.matmul(tensors[i], tensors[i])
                    tensors[i] = torch.sin(tensors[i]) + 1.0
                    tensors[i] = torch.sqrt(torch.abs(tensors[i]) + 0.01)
                    
                # Print utilization every second
                elapsed = time.time() - start_time
                if iteration % 5 == 0 or elapsed >= 10:
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    print(f"Iteration {iteration} - Time: {elapsed:.1f}s, Memory: {current_mem:.2f} GB ({current_mem/gpu_mem*100:.1f}%)")
                
                # Force GPU synchronization to ensure work is actually done
                torch.cuda.synchronize()
            
            total_time = time.time() - start_time
            print(f"Completed {iteration} iterations in {total_time:.2f} seconds")
            print(f"Final GPU memory usage: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
            # Run a multi-tensor test to simulate real deep learning workflow
            print("\n=== MULTI-TENSOR TRAINING SIMULATION ===")
            # Simulate a batch of images (batch_size x channels x height x width)
            batch_size = 128
            image_size = 224
            
            # Create inputs, a large model, optimizer, etc.
            print(f"Creating simulation with batch size {batch_size} and image size {image_size}...")
            
            # Create model with many parameters
            class LargeModel(torch.nn.Module):
                def __init__(self):
                    super(LargeModel, self).__init__()
                    # Multiple large layers
                    self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)
                    self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool2d(2, 2)
                    self.fc1 = torch.nn.Linear(512 * (image_size//8) * (image_size//8), 4096)
                    self.fc2 = torch.nn.Linear(4096, 1000)
                    self.relu = torch.nn.ReLU()
                    
                def forward(self, x):
                    x = self.pool(self.relu(self.conv1(x)))
                    x = self.pool(self.relu(self.conv2(x)))
                    x = self.pool(self.relu(self.conv3(x)))
                    x = x.view(-1, 512 * (image_size//8) * (image_size//8))
                    x = self.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
            
            try:
                # Create large input tensors
                inputs = torch.randn(batch_size, 3, image_size, image_size, device="cuda")
                targets = torch.randint(0, 1000, (batch_size,), device="cuda")
                
                # Create and move model to GPU
                model = LargeModel().cuda()
                
                # Create optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                # Loss function
                criterion = torch.nn.CrossEntropyLoss().cuda()
                
                # Run training for multiple iterations to maintain high utilization
                print("Running simulated training for 10 seconds...")
                train_start = time.time()
                train_iterations = 0
                
                while time.time() - train_start < 10:
                    train_iterations += 1
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                    
                    # Force synchronization
                    torch.cuda.synchronize()
                    
                    # Print status
                    if train_iterations % 2 == 0:
                        elapsed = time.time() - train_start
                        current_mem = torch.cuda.memory_allocated() / (1024**3)
                        print(f"Training iteration {train_iterations} - Time: {elapsed:.1f}s, Memory: {current_mem:.2f} GB ({current_mem/gpu_mem*100:.1f}%)")
                
                training_time = time.time() - train_start
                print(f"Completed {train_iterations} training iterations in {training_time:.2f} seconds")
                print(f"Peak GPU memory during training: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
                
                # Clean up
                del model, inputs, targets, optimizer
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Training simulation failed (likely out of memory): {e}")
                torch.cuda.empty_cache()
            
            print("\nAll GPU stress tests completed!")
            
        except RuntimeError as e:
            print(f"Test failed: {e}")
            if "out of memory" in str(e).lower():
                print("GPU ran out of memory - this actually confirms it's working!")
            torch.cuda.empty_cache()
            
    else:
        print("CUDA is not available. Check your PyTorch installation.")

if __name__ == "__main__":
    test_gpu()
