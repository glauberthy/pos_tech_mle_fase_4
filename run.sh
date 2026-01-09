#!/bin/bash

# 1. Inicia a API (FastAPI) em background na porta 8000
# O 'nohup' garante que ela não morra se o shell receber sinal
nohup uvicorn src.app:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

echo "Aguardando API iniciar..."
sleep 5

# 2. Inicia o Frontend (Streamlit) na porta 7860 (Porta oficial do HF Spaces)
# O Streamlit atua como proxy reverso visual para o usuário
streamlit run src/dashboard.py --server.port 7860 --server.address 0.0.0.0