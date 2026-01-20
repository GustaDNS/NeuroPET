import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBrainCNN(nn.Module):
    """
    Uma Rede Neural Convolucional simples para classificação binária.
    Entrada: Imagens 128x128 (Grascale - 1 canal)
    """
    def __init__(self):
        super(SimpleBrainCNN, self).__init__()
        # Camada Conv 1: 1 canal entrada -> 16 canais saída, kernel 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduz tamanho pela metade (128->64)
        
        # Camada Conv 2: 16 -> 32 canais
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Após pool 2x: imagem vira 32x32 com 32 canais
        
        # Camadas Fully Connected (Classificação)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2) # 2 classes: Normal vs Alzheimer

    def forward(self, x):
        # Feature Extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flattening
        x = x.view(-1, 32 * 32 * 32)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Nota: Não usamos Softmax aqui porque usaremos CrossEntropyLoss no treino
        return x
