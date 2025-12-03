# 😃 Sistema de Detecção de Expressões Faciais

Este projeto implementa um sistema de detecção de expressões faciais em tempo real utilizando técnicas de **Visão Computacional** e **Processamento de Imagens**.  
A solução foi desenvolvida com **Python**, fazendo uso das bibliotecas **MediaPipe**, **OpenCV** e **NumPy** para rastreamento facial, cálculo geométrico entre landmarks e renderização de imagens.

O objetivo é classificar expressões faciais simples com base em proporções geométricas do rosto e exibir uma imagem correspondente à expressão detectada.

---

## 🧠 Funcionalidades

✔️ Detecção facial em tempo real usando MediaPipe  
✔️ Identificação de cinco expressões:
- **Sorriso** → mostra `smile.jpg`
- **Dedo no rosto** → mostra `touch.png` 
- **Mãos levantadas** → mostra `air.jpg`
- **Olhos arregalados** → mostra `scary.png`
- **Expressão neutra** → mostra `plain.png`

✔️ Renderização automática da imagem associada à expressão  
✔️ Lógica baseada em distâncias entre landmarks faciais  
✔️ Código simples, rápido e eficiente

---

## 🚀 Tecnologias utilizadas

- **Python 3.10**
- **OpenCV**
- **MediaPipe**
- **Numpy**

---

## 📦 Instalação

### 1️⃣ Clone o repositório

```bash
https://github.com/Gledson-Perdival-Junior/sistema_de_reconhecimento_facial.git
cd sistema_de_reconhecimento_facial
```

### 2️⃣ Crie um ambiente virtual
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
```

### 3️⃣ Instale as dependências
```bash
pip install opencv-python mediapipe numpy
```

### ▶️ Como executar
```bash
python reconhecimento.py
```


O sistema abrirá a webcam e detectará expressões faciais automaticamente.

## 🧩 Como funciona a detecção?

O algoritmo utiliza landmarks da malha facial do MediaPipe.
A partir deles, calcula:

Distância entre os cantos da boca → detectar sorriso

Distância entre o dedo e rosto → detectar toque

Presença da mão na imagem → detectar mão

Abertura entre pálpebra superior e inferior → detectar olhos arregalados

Com base nesses valores, o código decide qual imagem deve ser exibida.
