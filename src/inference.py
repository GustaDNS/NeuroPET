import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
# Ajuste de path para importar o modelo se rodado do root
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import SimpleBrainCNN

class NeuroPredictor:
    def __init__(self, model_path='models/neuropet_cnn.pth'):
        self.device = torch.device("cpu") # Inferência na CPU é ok para 1 imagem
        self.model = SimpleBrainCNN().to(self.device)
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
             
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Modo de avaliação
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Mapeamento invertido (índice -> nome da classe)
        # Assumindo que Alzheimer=0, Normal=1 baseado na ordem alfabética da pasta
        self.idx_to_class = {0: 'Alzheimer Disease', 1: 'Normal Control'}

    def generate_heatmap(self, image_tensor):
        """
        Gera um mapa de saliência simples (onde os gradientes são mais altos).
        Uma forma básica de "Explainable AI".
        """
        image_tensor.requires_grad_()
        output = self.model(image_tensor)
        score, predicted_idx = torch.max(output, 1)
        score.backward()
        
        # Pega o gradiente máximo absoluto em todos os canais para cada pixel
        saliency, _ = torch.max(image_tensor.grad.data.abs(), dim=1)
        saliency = saliency.reshape(128, 128)
        
        # Normaliza entre 0 e 1 para visualização
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return saliency.numpy()

    def predict(self, image_path):
        """Recebe o caminho de uma imagem e retorna a predição e o heatmap."""
        img_pil = Image.open(image_path)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # 1. Predição
        outputs = self.model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        predicted_label = self.idx_to_class[predicted_idx.item()]
        
        # 2. Gerar Heatmap Explicativo
        heatmap = self.generate_heatmap(img_tensor)

        return {
            "prediction": predicted_label,
            "confidence": confidence.item(),
            "heatmap": heatmap,
            "original_image": img_pil
        }
