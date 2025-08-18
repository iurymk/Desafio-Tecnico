# Diretório Models

Este diretório contém os modelos de machine learning treinados e arquivos relacionados.

## Estrutura

```
models/
├── README.md                    # Este arquivo
├── best_model.pkl              # Modelo treinado final (será gerado)
├── preprocessors.pkl           # Preprocessadores (será gerado)
├── metadata.json              # Metadados do modelo (será gerado)
└── .gitkeep                   # Mantém o diretório no git
```

## Arquivos

### best_model.pkl
- Modelo de machine learning com melhor performance
- Formato: pickle (.pkl)
- Gerado automaticamente pelo `model_training.py`

### preprocessors.pkl
- Scaler para normalização
- Label encoders para variáveis categóricas
- Feature selector
- Essenciais para aplicar o modelo em novos dados

### metadata.json
- Performance do modelo (AUC, F1-score, accuracy)
- Hiperparâmetros utilizados
- Data de treinamento
- Lista de features utilizadas

## Como usar

```python
import pickle
import json

# Carregar modelo
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Carregar preprocessadores
with open('models/preprocessors.pkl', 'rb') as f:
    preprocessors = pickle.load(f)

# Carregar metadados
with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

# Fazer predições
# (exemplo de uso)
```

## Geração

Os arquivos são gerados automaticamente executando:
```bash
python src/model_training.py
```
