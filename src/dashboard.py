import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os

# Configuração da Página
st.set_page_config(
    page_title="Petrobras AI Forecast",
    page_icon="📈",
    layout="wide"
)

# --- CONFIGURAÇÃO DINÂMICA DE URL ---
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("📈 Tech Challenge Fase 4: Petrobras Forecast")
st.markdown("---")

# Sidebar
st.sidebar.header("Ambiente")
status_color = "🟢" if "localhost" in API_URL else "🔵"
st.sidebar.markdown(f"**Conectado em:** `{API_URL}` {status_color}")

st.sidebar.header("Navegação")
page = st.sidebar.radio("Ir para:", ["🔮 Simulador de Previsão", "📊 Monitoramento (Ops)"])

def get_metrics():
    try:
        response = requests.get(f"{API_URL}/metrics")
        return response.text
    except:
        return None

def extract_metric_value(text, metric_name):
    """
    Função Robustecida:
    1. Procura todas as linhas que começam com o nome da métrica.
    2. Pega o último valor da linha (evita problemas com labels {}).
    3. Soma tudo (ex: soma requisições do /predict + /metrics + /health).
    """
    total = 0.0
    if text:
        for line in text.split('\n'):
            # Ignora comentários
            if line.startswith('#'): continue
            
            # Se a linha começa com o nome da métrica
            if line.startswith(metric_name):
                try:
                    # O valor numérico é sempre a última parte da string no formato Prometheus
                    # Ex: http_requests_total{method="post"} 15.0
                    value = float(line.split()[-1])
                    total += value
                except:
                    pass
    return total

# --- PÁGINA 1: SIMULADOR ---
if page == "🔮 Simulador de Previsão":
    st.header("Simulador de Inferência (LSTM)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.info("O modelo utiliza os últimos 60 dias para prever o próximo fechamento.")
        if st.button("🔄 Carregar Dados Reais (Yahoo/Cache)", use_container_width=True):
            with st.spinner("Buscando dados na API..."):
                try:
                    resp = requests.get(f"{API_URL}/sample-data")
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state['input_data'] = data['last_60_days']
                        st.session_state['source'] = data.get('source', 'Unknown')
                        st.success(f"Dados carregados! Fonte: {st.session_state['source']}")
                    else:
                        st.error("Erro ao buscar dados.")
                except Exception as e:
                    st.error(f"API fora do ar: {e}")

    if 'input_data' in st.session_state:
        prices = st.session_state['input_data']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prices, mode='lines+markers', name='Histórico (60 dias)'))
        fig.update_layout(title="Janela de Entrada", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🚀 Realizar Previsão", type="primary", use_container_width=True):
            with st.spinner("Processando na Rede Neural..."):
                payload = {"last_60_days": prices}
                try:
                    resp_pred = requests.post(f"{API_URL}/predict", json=payload)
                    if resp_pred.status_code == 200:
                        result = resp_pred.json()
                        pred_price = result['predicted_price_brl']
                        st.metric(label="Preço Previsto (D+1)", value=f"R$ {pred_price:.2f}")
                        
                        last_day = len(prices)
                        fig.add_trace(go.Scatter(
                            x=[last_day], 
                            y=[pred_price], 
                            mode='markers+text',
                            marker=dict(color='red', size=15),
                            text=[f"R$ {pred_price}"],
                            textposition="top center",
                            name='Previsão'
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Erro na previsão: {resp_pred.text}")
                except Exception as e:
                    st.error(f"Erro de conexão: {e}")

# --- PÁGINA 2: MONITORAMENTO ---
elif page == "📊 Monitoramento (Ops)":
    st.header("Monitoramento de Performance da API")
    
    if st.button("Atualizar Métricas"):
        st.rerun()
        
    raw_metrics = get_metrics()
    
    if raw_metrics:
        # 1. Total de Requisições (Soma todas as rotas)
        total_req = extract_metric_value(raw_metrics, 'http_requests_total')
        
        # 2. Latência (Soma dos tempos / Soma das contagens)
        # Usamos 'highr' para alta precisão ou 'seconds' padrão
        sum_lat = extract_metric_value(raw_metrics, 'http_request_duration_highr_seconds_sum')
        count_lat = extract_metric_value(raw_metrics, 'http_request_duration_highr_seconds_count')
        
        # Evita divisão por zero
        avg_latency = (sum_lat / count_lat) if count_lat > 0 else 0
        
        # 3. Métricas de Sistema (CPU e RAM)
        cpu_usage = extract_metric_value(raw_metrics, 'process_cpu_seconds_total')
        
        # RAM vem em Bytes -> Converter para MB
        mem_usage = extract_metric_value(raw_metrics, 'process_resident_memory_bytes') / 1024 / 1024 
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requisições Totais", f"{int(total_req)}")
        c2.metric("Latência Média", f"{avg_latency:.4f} s")
        c3.metric("CPU Time (s)", f"{cpu_usage:.2f}")
        c4.metric("Uso de RAM", f"{mem_usage:.1f} MB")
        
        st.subheader("Log Bruto do Prometheus")
        with st.expander("Ver detalhes técnicos"):
            st.code(raw_metrics)
    else:
        st.warning("⚠️ Não foi possível conectar ao endpoint /metrics")