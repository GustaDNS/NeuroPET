FROM python:3.9-slim

WORKDIR /app

# Instala dependências do sistema para processamento de imagem
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY requirements.txt .
# Instalação do PyTorch versão CPU para o container ficar leve (em produção usaria GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gera dados e treina o modelo na construção da imagem
RUN python3 src/data_gen.py && python3 src/train.py

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--theme.base=dark"]
