# 1. Imagem Base: Python 3.10 leve
FROM python:3.10-slim

# 2. Define diretório de trabalho dentro do container
WORKDIR /app

# 3. Copia e instala as dependências (Otimização de cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia o código fonte e os modelos treinados
COPY src/ ./src/
COPY models/ ./models/

# 5. Exposição da porta (Documentação)
EXPOSE 8000

# 6. Variáveis de ambiente para o TensorFlow (Desativar GPU no Docker)
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="2"

# 7. Comando para rodar a API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]