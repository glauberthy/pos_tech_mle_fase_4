FROM python:3.11-slim

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia código
COPY src/ ./src/
COPY models/ ./models/
COPY run.sh .
RUN chmod +x run.sh

# Define a URL padrão da API interna
ENV API_URL="http://localhost:8000"

EXPOSE 7860

CMD ["./run.sh"]