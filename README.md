# 🧠 NeuroPET: AI-Powered Alzheimer's Diagnostics

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

![Project Banner](banner.png)

> **A Deep Learning diagnostic support tool for detecting Alzheimer's disease signatures in PET scan imagery, featuring Explainable AI (XAI) and a modern dark-mode interface.**

---


## 📋 Table of Contents
- [📍 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture & Tech Stack](#%EF%B8%8F-architecture--tech-stack)
- [🚀 Getting Started](#-getting-started)
- [🤝 Contributing](#-contributing)

---

## 📍 Overview

**NeuroPET** is an end-to-end Data Science project designed to assist medical professionals in analyzing brain PET (Positron Emission Tomography) scans.

Alzheimer's disease is often characterized by reduced glucose metabolism in specific brain regions (hypometabolism). NeuroPET uses a **Convolutional Neural Network (CNN)** built with **PyTorch** to analyze axial slices of PET scans and classify them as "Normal Control" or "Alzheimer's Disease".

Crucially, it addresses the "black box" problem in AI by providing **visual explainability**, overlaying heatmaps on the scans to show exactly where the model focused to make its prediction.

---

## ✨ Key Features

* **🧠 Deep Learning Classifier:** A custom CNN trained on synthetic PET data to detect metabolic anomalies.
* **👁️ Explainable AI (XAI):** Generates saliency heatmaps (using gradient-based techniques) to visualize model attention spots, increasing clinical trust.
* **🖥️ Modern Streamlit Dashboard:** A polished, dark-mode interface with glassmorphism effects and advanced data visualization using Matplotlib.
* **⚙️ Real-Time System Monitoring:** Integrated hardware tracking displays active RAM and GPU usage during inference using `psutil` and `torch`.
* **📂 End-to-End Pipeline:** Includes scripts for synthetic data generation, model training, and model inference.

---

## 🏗️ Architecture & Tech Stack

The project follows a modular structure separating data generation, training logic, and the frontend application.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Core AI** | **PyTorch** | Construction and training of the Convolutional Neural Network (CNN). |
| **Frontend** | **Streamlit** | Interactive web dashboard for image upload and result visualization. |
| **Data Processing** | **NumPy / Scikit-image** | Generation of synthetic medical imagery and tensor transformations. |
| **Visualization** | **Matplotlib / PIL** | Rendering advanced side-by-side comparison plots and heatmaps. |
| **Utilities** | **psutil** | Real-time system resource monitoring in the sidebar. |
| **Containerization** | **Docker** | (Optional) Ensures reproducibility across environments. |

### CNN Structure (Simplified):
`Input (128x128x1) -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> FC Linear -> Output (2 Classes)`

---

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
* Python 3.8+
* Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/GustaDNS/NeuroPET.git](https://github.com/GustaDNS/Neuro_PET.git)
    cd NeuroPET
    ```

2.  **Set up Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    *Note: PyTorch installation depends on your hardware (CPU vs CUDA). The command below is for CPU-only for broad compatibility.*
    ```bash
    pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    pip install -r requirements.txt
    ```

### Running the Pipeline

Since we cannot use real patient data due to privacy regulations, this project includes a generator for synthetic PET slices.

1.  **Generate Synthetic Data:**
    ```bash
    python src/data_gen.py
    ```
    *(Creates dummy images in `data/raw/`)*

2.  **Train the CNN Model:**
    ```bash
    python src/train.py
    ```
    *(Trains for 5 epochs and saves weights to `models/neuropet_cnn.pth`)*

3.  **Launch the Dashboard:**
    ```bash
    streamlit run app/dashboard.py
    ```
    The app will open in your browser at `http://localhost:8501`.

---

## 🤝 Contributing

Contributions are welcome! This is a portfolio project intended for educational and demonstration purposes.

1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

*Disclaimer: NeuroPET is a demonstration tool using synthetic data and simple CNN architectures. It is NOT intended for actual clinical diagnosis.*
