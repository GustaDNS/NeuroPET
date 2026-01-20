import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from model import SimpleBrainCNN

def train_model(epochs=5):
    print("⚙️ Starting Deep Learning Training (this may take a moment)...")
    
    if not os.path.exists('data/raw/Normal'):
        raise FileNotFoundError("Data not found. Run src/data_gen.py first.")

    # 1. Setup de Dados e Transformações
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # Converte imagem [0-255] para tensor [0-1]
        transforms.Normalize((0.5,), (0.5,)) # Normaliza para [-1, 1]
    ])

    full_dataset = datasets.ImageFolder(root='data/raw', transform=transform)
    
    # Split Treino/Validação
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 2. Inicializa Modelo, Loss e Otimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    model = SimpleBrainCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Loop de Treinamento
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # 4. Salvar Modelo
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/neuropet_cnn.pth')
    # Salva o mapeamento de classes também
    with open('models/class_indices.txt', 'w') as f:
        f.write(str(full_dataset.class_to_idx))
        
    print("💾 PyTorch model saved to models/neuropet_cnn.pth")

if __name__ == "__main__":
    train_model()
