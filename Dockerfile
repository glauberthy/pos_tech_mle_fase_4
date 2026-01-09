# Use uma imagem base oficial e leve
FROM python:3.11-slim

WORKDIR /app

# Variáveis de ambiente para evitar arquivos .pyc e logs de buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_URL="http://localhost:8000"

# Instala dependências do sistema necessárias para compilação (se houver)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python (Cache Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY src/ ./src/
COPY run.sh .

# Cria diretório de modelos se não existir e ajusta permissões
RUN mkdir -p src/models && chmod +x run.sh

# Garante que o usuário do HF (1000) tenha permissão na pasta
RUN chown -R 1000:1000 /app

# Troca para usuário não-root (Segurança e boa prática HF)
USER 1000

# Expõe a porta padrão do HF Spaces
EXPOSE 7860

# Comando de entrada
CMD ["./run.sh"]