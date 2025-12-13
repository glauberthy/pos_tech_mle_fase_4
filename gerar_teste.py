import yfinance as yf
import json
import pandas as pd
from datetime import datetime, timedelta

# --- Configuração ---
SYMBOL = 'PETR4.SA'
# Data alvo para a previsão (o script pegará os 60 dias ANTERIORES a isso)
TARGET_DATE = '2025-12-13' 
API_URL = 'http://localhost:8000/predict'

def gerar_curl_pronto():
    print(f"⏳ Baixando histórico recente da {SYMBOL} no Yahoo Finance...")
    
    # 1. Baixar dados com margem de segurança
    start_date = (datetime.strptime(TARGET_DATE, "%Y-%m-%d") - timedelta(days=150)).strftime("%Y-%m-%d")
    
    # Baixa os dados
    df = yf.download(SYMBOL, start=start_date, end=TARGET_DATE, progress=True, auto_adjust=True)
    
    # Tratamento para multi-index do yfinance (versões novas)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(SYMBOL, level=1, axis=1)
        except:
            pass 
            
    # Limpeza
    df['Close'] = df['Close'].ffill()
    
    # 2. Pegar os últimos 60 dias exatos
    last_60 = df['Close'].tail(60)
    
    if len(last_60) < 60:
        print(f"❌ Erro: Histórico insuficiente. Apenas {len(last_60)} dias encontrados.")
        return

    # 3. Criar a lista de valores
    valores = last_60.values.flatten().tolist()
    valores = [round(float(x), 2) for x in valores] # Garante float e arredonda

    # 4. Montar o JSON
    payload = {
        "last_60_days": valores
    }
    json_body = json.dumps(payload)

    # 5. Montar o comando CURL
    # Nota: Usamos aspas simples ' ao redor do corpo para evitar problemas no Linux/Mac
    curl_command = (
        f"curl -X 'POST' \\\n"
        f"  '{API_URL}' \\\n"
        f"  -H 'accept: application/json' \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{json_body}'"
    )

    print("\n" + "="*80)
    print(f"🚀 COMANDO CURL GERADO (Dados de {last_60.index[0].date()} a {last_60.index[-1].date()})")
    print("="*80 + "\n")
    print(curl_command)
    print("\n" + "="*80 + "\n")

    # 6. Tentar mostrar o GABARITO (Preço Real do dia seguinte)
    try:
        # Pega dados a partir da data alvo
        real_df = yf.download(SYMBOL, start=TARGET_DATE, period='5d', progress=True, auto_adjust=True)
        
        # Novamente, trata multi-index se necessário
        if isinstance(real_df.columns, pd.MultiIndex):
             real_df = real_df.xs(SYMBOL, level=1, axis=1)
             
        if not real_df.empty:
            # O primeiro registro é o dia alvo real
            price_real = real_df['Close'].iloc[0]
            data_real = real_df.index[0].date()
            print(f"🎯 GABARITO DO MERCADO:")
            print(f"   Data Real: {data_real}")
            print(f"   Preço Real de Fechamento: R$ {price_real:.2f}")
            print("   (Compare este valor com o 'predicted_price_brl' que a API retornar!)")
        else:
            print("ℹ️  Sem dados futuros disponíveis (você está testando com a data de hoje/futuro).")
    except Exception as e:
        print(f"⚠️ Não foi possível buscar o gabarito: {e}")

if __name__ == "__main__":
    gerar_curl_pronto()