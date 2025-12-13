# 📈 Tech Challenge Fase 4 - Previsão de Ações com LSTM

Projeto de conclusão da Fase 4 da Pós-Graduação em Machine Learning Engineering. O objetivo é desenvolver um modelo de Deep Learning (LSTM) para prever o preço de fechamento de ações (PETR4) e realizar o deploy em produção.

## 🚀 Funcionalidades

- **Coleta Automática:** Script para download e tratamento de dados do Yahoo Finance (`yfinance`).
- **Deep Learning:** Modelo LSTM (Long Short-Term Memory) treinado com histórico de 2018-2024.
- **API RESTful:** Desenvolvida com **FastAPI** para inferência em tempo real.
- **Containerização:** Aplicação empacotada com **Docker** para execução em qualquer ambiente.
- **Métricas de Performance:** Monitoramento de MAE, RMSE e MAPE.

## 🛠️ Arquitetura

O projeto segue uma arquitetura modular:
1.  **Data Ingestion:** Extração e limpeza (`src/coleta_dados.py`).
2.  **Training:** Notebook de treino (`notebooks/01_treinamento_lstm.ipynb`) que gera os artefatos.
3.  **Inference:** API (`src/app.py`) carrega o modelo `.keras` e o scaler `.pkl` no startup.

## 📦 Como Rodar (Docker)

Esta é a forma recomendada de execução.

### 1. Construir a Imagem
```bash
docker build -t tech-challenge-fase4 .
````

### 2\. Rodar o Container

```bash
docker run -p 8000:8000 tech-challenge-fase4
```

A API estará disponível em: `http://localhost:8000`

## 📚 Documentação da API

Acesse o Swagger UI para testar os endpoints interativamente:
👉 **[http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)**

### Exemplo de Requisição (CURL)

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{ "last_60_days": [30.5, 31.2, ... (lista com 60 preços)] }'
```

## 📊 Resultados do Modelo

O modelo foi validado com dados de teste (20% do dataset), obtendo:

  - **MAPE (Erro Percentual):** \~2.25%
  - **RMSE:** \~0.88
