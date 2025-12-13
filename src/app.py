import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime, timedelta

# --- Configuração de Logs (Requisito de Monitoramento) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API_Petrobras")

FALLBACK_DATA = [
    31.52, 31.64, 31.37, 31.04, 31.33, 31.95, 32.54, 32.32, 32.25, 31.84,
    31.46, 31.44, 31.08, 30.73, 29.93, 30.21, 30.07, 29.73, 29.41, 29.71,
    29.77, 29.53, 29.88, 30.16, 29.83, 30.00, 30.01, 30.05, 29.93, 29.77,
    30.10, 30.25, 30.85, 31.01, 32.18, 32.36, 33.20, 32.35, 32.49, 32.70,
    32.88, 32.99, 32.82, 32.57, 32.54, 32.28, 32.23, 32.40, 31.79, 31.85,
    32.07, 32.31, 32.52, 31.37, 31.66, 31.86, 31.94, 31.26, 31.41, 31.26
]

# --- Variáveis Globais ---
# O modelo e o scaler ficam na memória RAM para acesso rápido
ml_models = {}

# --- Ciclo de Vida (Lifespan) ---
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

# --- Inicialização do App ---
app = FastAPI(
    title="Tech Challenge Fase 4 - Forecast API",
    description="API para previsão de preço de ações (PETR4) usando LSTM.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Contrato de Dados (Input Schema) ---
class StockInput(BaseModel):
    last_60_days: list[float] = Field(
        ..., 
        min_length=60, 
        max_length=60, 
        description="Lista exata de 60 preços de fechamento anteriores.",
        example=[30.0 + (i * 0.1) for i in range(60)] #fake 
    )

# ---Rotas ---

@app.get("/health")
def health_check():
    """Verifica se a API e o modelo estão saudáveis"""
    if 'error' in ml_models:
        return {"status": "error", "detail": ml_models['error']}
    return {
        "status": "healthy", 
        "model_loaded": 'model' in ml_models and 'scaler' in ml_models
    }

@app.get("/sample-data")
def get_sample_data():
    """
    Tenta buscar dados reais no Yahoo Finance.
    Se falhar (bloqueio/timeout), retorna dados de fallback para garantir o teste.
    """
    symbol = 'PETR4.SA'
    
    try:
        logger.info(f"Tentando buscar dados ao vivo para {symbol}...")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
        
        # Tenta baixar
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError("Yahoo retornou vazio.")

        # Tratamento MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=1, axis=1)
            
        last_60 = df['Close'].ffill().tail(60)
        
        if len(last_60) < 60:
            raise ValueError(f"Dados insuficientes: {len(last_60)}")
            
        values = [round(float(x), 2) for x in last_60.values.flatten().tolist()]
        
        logger.info("✅ Dados ao vivo obtidos com sucesso.")
        return {
            "source": "yahoo_finance_live",
            "last_60_days": values
        }
        
    except Exception as e:
        # CIRCUIT BREAKER: Se der erro, usa o fallback
        logger.warning(f"⚠️ Falha no Yahoo Finance ({e}). Usando dados de Fallback.")
        return {
            "source": "fallback_cached_data",
            "note": "Yahoo Finance indisponível no momento. Usando dados recentes em cache.",
            "last_60_days": FALLBACK_DATA
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