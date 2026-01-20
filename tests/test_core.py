import pytest
import torch
import os
import sys
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleBrainCNN
from src.inference import NeuroPredictor

def test_model_architecture():
    """Verifica se a CNN produz a saída com formato correto (2 classes)"""
    model = SimpleBrainCNN()
    # Cria uma imagem falsa (batch_size=1, canais=1, 128x128)
    fake_input = torch.randn(1, 1, 128, 128)
    output = model(fake_input)
    assert output.shape == (1, 2), "O output do modelo deve ser [batch_size, 2 classes]"

def test_inference_flow():
    """Testa o fluxo completo: carrega modelo, processa imagem, prevê"""
    # Precisamos de uma imagem dummy e um modelo treinado
    if not os.path.exists('models/neuropet_cnn.pth'):
        pytest.skip("Modelo não treinado, pulando teste de inferência.")
    
    dummy_img_path = "tests/dummy_brain.png"
    # Cria uma imagem preta dummy
    img = Image.fromarray(np.zeros((128, 128), dtype='uint8'), 'L')
    img.save(dummy_img_path)
    
    try:
        predictor = NeuroPredictor()
        result = predictor.predict(dummy_img_path)
        assert "prediction" in result
        assert "heatmap" in result
        assert 0.0 <= result['confidence'] <= 1.0
    finally:
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)
