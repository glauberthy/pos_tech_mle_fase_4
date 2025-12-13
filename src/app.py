import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging
import os

# --- 1. Configuração de Logs (Requisito de Monitoramento) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API_Petrobras")

# --- 2. Variáveis Globais ---
# O modelo e o scaler ficam na memória RAM para acesso rápido
ml_models = {}

# --- 3. Ciclo de Vida (Lifespan) ---
# Executa apenas UMA vez quando o servidor sobe
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("🚀 Iniciando carregamento do modelo LSTM...")
        
        # Caminhos relativos à raiz do projeto
        model_path = 'models/lstm_model.keras'
        scaler_path = 'models/scaler.pkl'

        # Verificar se arquivos existem
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Arquivos não encontrados em {os.getcwd()}. Verifique a pasta 'models'.")

        # Carregar artefatos
        ml_models['model'] = tf.keras.models.load_model(model_path)
        ml_models['scaler'] = joblib.load(scaler_path)
        
        logger.info("✅ Modelo e Scaler carregados com sucesso! API pronta.")
        yield
    except Exception as e:
        logger.error(f"❌ Erro crítico no startup: {e}")
        # Não impede a API de subir, mas avisa o erro
        ml_models['error'] = str(e)
        yield
    finally:
        logger.info("🛑 Desligando API e liberando recursos.")

# --- 4. Inicialização do App ---
app = FastAPI(
    title="Tech Challenge Fase 4 - Forecast API",
    description="API para previsão de preço de ações (PETR4) usando LSTM.",
    version="1.0.0",
    lifespan=lifespan
)

# --- 5. Contrato de Dados (Input Schema) ---
class StockInput(BaseModel):
    last_60_days: list[float] = Field(
        ..., 
        min_length=60, 
        max_length=60, 
        description="Lista exata de 60 preços de fechamento anteriores.",
        example=[30.0 + (i * 0.1) for i in range(60)] #fake 
    )

# --- 6. Rotas ---

@app.get("/health")
def health_check():
    """Verifica se a API e o modelo estão saudáveis"""
    if 'error' in ml_models:
        return {"status": "error", "detail": ml_models['error']}
    return {
        "status": "healthy", 
        "model_loaded": 'model' in ml_models and 'scaler' in ml_models
    }

@app.post("/predict")
def predict_price(input_data: StockInput):
    """Recebe 60 dias de histórico e prevê o próximo dia"""
    
    # Validação de segurança
    if 'model' not in ml_models or 'scaler' not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo não está disponível no servidor.")

    try:
        # A. Preparar dados
        features = np.array(input_data.last_60_days).reshape(-1, 1) # (60, 1)
        
        # B. Normalizar (Usando o scaler treinado)
        scaled_features = ml_models['scaler'].transform(features)
        
        # C. Formatar para LSTM (Batch, Steps, Features) -> (1, 60, 1)
        final_input = scaled_features.reshape(1, 60, 1)
        
        # D. Previsão
        prediction_scaled = ml_models['model'].predict(final_input, verbose=0)
        
        # E. Desnormalizar (Voltar para R$)
        prediction_real = ml_models['scaler'].inverse_transform(prediction_scaled)
        result = float(prediction_real[0][0])
        
        logger.info(f"🔮 Previsão solicitada. Resultado: R$ {result:.2f}")
        
        return {
            "ticker": "PETR4.SA",
            "predicted_price_brl": round(result, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Erro na inferência: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no processamento do modelo.")