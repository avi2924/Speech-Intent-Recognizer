import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAudioGRU(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(CNNAudioGRU, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Calculate GRU input size based on CNN output
        # This depends on your mel spectrogram dimensions and pooling
        # For typical 64x200 input with 3 max pooling layers (2x2), the output would be
        # channels: 128, height: 8, width: 25
        # So self.gru_input_size = 128 * 8 = 1024
        self.gru_input_size = 1024  # This must match the actual flattened CNN output
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=self.gru_input_size,  # Changed from 128 to 1024
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.Linear(512, 1)  # 512 because bidirectional (256*2)
        
        # Output layer
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # Apply convolutions with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Reshape for GRU - this is where the mismatch occurred
        b, c, h, w = x.size()
        # Preserve the batch and time dimensions, but flatten the channels and frequency
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c*h)
        
        # Apply dropout
        x = self.dropout(x)
        
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
