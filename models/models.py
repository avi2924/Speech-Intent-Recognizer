import torch
import torch.nn as nn

class CNNAudioGRU(nn.Module):
    def __init__(self, input_dim=64, num_classes=31, hidden_size=128, num_layers=2, dropout=0.3):
        super(CNNAudioGRU, self).__init__()

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Compute the GRU input size after CNN layers
        self.flattened_cnn_dim = (input_dim // 4) * (32)  # 2 pool layers reduce dim by 4

        # RNN for temporal modeling
        self.rnn = nn.GRU(
            input_size=self.flattened_cnn_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [B, mel_bins, time]
        x = x.unsqueeze(1)  # Add channel dim: [B, 1, mel_bins, time]
        x = self.cnn(x)     # Output: [B, C, mel_bins', time']

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, time, C, mel]
        x = x.view(b, w, -1)  # [B, time, features]

        out, _ = self.rnn(x)  # [B, time, hidden*2]
        out = out[:, -1, :]   # Take last time step

        out = self.classifier(out)
        return out

if __name__ == '__main__':
    model = CNNAudioGRU()
    dummy_input = torch.randn(4, 64, 200)  # [batch, mel, time]
    output = model(dummy_input)
    print("Output shape:", output.shape)  # [4, num_classes]
