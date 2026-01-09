import yfinance as yf
import pandas as pd
import os

# Configurações iniciais baseadas no PDF [cite: 21, 22, 23]
SYMBOL = 'PETR4.SA' 
START_DATE = '2018-01-01'
END_DATE = '2024-12-30' # Atualizado para pegar dados recentes

def download_data():
    print(f"Baixando dados para {SYMBOL}...")
    
    #Use a função download para obter os dados
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    
    # Verificação básica de qualidade (Issue #2 simplificada)
    if df.empty:
        print("Erro: Nenhum dado baixado. Verifique o Símbolo ou sua internet.")
        return
    
    # Tratamento de nulos com ffill (preencher gaps)
    df = df.ffill()
    
    # Salvar em CSV para inspeção fácil (mais simples que Parquet agora)
    os.makedirs('data', exist_ok=True)
    file_path = f'data/{SYMBOL}.csv'
    df.to_csv(file_path)
    
    print(f"Sucesso! {len(df)} linhas salvas em {file_path}")
    print(df.head())

if __name__ == "__main__":
    download_data()