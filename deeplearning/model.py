import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3):
        super(TransformerPredictor, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # TransformerEncoder 需要 [seq_len, batch, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # 恢复 [batch, seq_len, d_model]
        return self.fc(x[:, -1, :])


if __name__ == "__main__":
    model = TransformerPredictor(input_dim=5)
    print(model)
