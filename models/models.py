import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAudioGRU(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(CNNAudioGRU, self).__init__()
        
        # CNN layers - GPU optimized with bias=False for better performance
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Use inplace operations where possible for better memory efficiency
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        
        # GRU input size
        self.gru_input_size = 1024
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Attention mechanism
        self.attention = nn.Linear(512, 1)
        
        # Final classification layer
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Ensure input has right shape for GPU
        batch_size = x.size(0)
        
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # CNN feature extraction with inplace operations
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Reshape for GRU - use contiguous for better memory layout
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c*h)
        
        # Apply GRU
        x, _ = self.gru(x)
        
        # Apply attention
        attn_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        
        # Final classification
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = CNNAudioGRU(num_classes=31)
    dummy_input = torch.randn(4, 64, 200)  # [batch, mel, time]
    output = model(dummy_input)
    print("Output shape:", output.shape)  # [4, num_classes]
