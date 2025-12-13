#!/bin/bash

# Inicia a API em segundo plano (&) na porta 8000
uvicorn src.app:app --host 0.0.0.0 --port 8000 &

# Aguarda 5 segundos para a API subir
sleep 5

# Inicia o Streamlit na frente, na porta 8501
streamlit run src/dashboard.py --server.port 7860 --server.address 0.0.0.0