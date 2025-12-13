# 📈 Tech Challenge Fase 4: Previsão de Ativos com LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)

Projeto de conclusão da Fase 4 da Pós-Graduação em Machine Learning Engineering.
Este projeto implementa um pipeline MLOps completo: desde a coleta de dados financeiros até o deploy de um modelo de Deep Learning em produção via container Docker.

---

## 🏛️ Arquitetura da Solução

O projeto foi desenhado seguindo princípios de microsserviços e reprodutibilidade. Abaixo, o fluxo de dados da solução:

```mermaid
graph LR
    A[Yahoo Finance] -->|Coleta de Dados| B(Data Cleaning)
    B -->|Normalização| C(Feature Engineering)
    C -->|Treinamento| D{Modelo LSTM}
    D -->|Artefatos| E[lstm_model.keras]
    D -->|Artefatos| F[scaler.pkl]
    
    User((Usuário)) -->|JSON: 60 dias| API[FastAPI]
    API -->|Carrega| E
    API -->|Carrega| F
    API -->|Inferência| User
````

## 📂 Estrutura do Projeto

A organização de diretórios segue o padrão funcional para separação de responsabilidades, garantindo modularidade e fácil manutenção:

```text
.
├── 📜 coleta_dados.py          # 📥 Script ETL para download e limpeza inicial
├── 🛠️ gerar_teste.py           # 🧪 Utilitário para gerar massa de teste (JSON) real
├── 🐳 Dockerfile               # 📦 Receita para containerização da aplicação
├── 📋 requirements.txt         # 📚 Dependências e bibliotecas do projeto
├── 📖 README.md                # 📄 Documentação Técnica
├── 📂 data/                    # 💾 Armazenamento de dados brutos (ignorados no git)
│   └── 📉 PETR4.SA.csv
├── 📂 models/                  # 🧠 Artefatos binários do modelo treinado
│   ├── 🤖 lstm_model.keras     # O modelo de Rede Neural serializado
│   └── 📏 scaler.pkl           # Objeto de normalização (MinMax)
├── 📂 notebooks/               # 🔬 Laboratório de Experimentação
│   └── 📓 01_treinamento_lstm.ipynb  # Notebook Principal (EDA, Treino e Validação)
└── 📂 src/                     # 🚀 Código Fonte da Aplicação (Produção)
    └── ⚡ app.py               # API RESTful de alta performance com FastAPI
````

-----

## 🧠 Decisões Técnicas

### 1\. Modelo: LSTM (Long Short-Term Memory)

Em conformidade com os requisitos mandatórios do **Tech Challenge Fase 4**, implementamos uma arquitetura baseada em **Redes Neurais Recorrentes (LSTM)**.

Esta escolha arquitetural é ideal para o problema proposto, pois as LSTMs superam modelos lineares tradicionais na captura de dependências de longo prazo e padrões não-lineares em séries temporais financeiras.

  * **Input:** Janela deslizante (*sliding window*) de **60 dias**.
  * **Métricas de Avaliação:** O modelo atingiu um **MAPE (Erro Percentual Absoluto Médio)** de **\~2.25%**, validando sua capacidade preditiva sobre a tendência do ativo.

### 2\. Backend: FastAPI

Utilizado em substituição ao Flask por ser assíncrono (ASGI), o que permite maior performance em inferência, além de gerar automaticamente a documentação Swagger/OpenAPI necessária para os testes da banca.

### 3\. Deploy: Docker

A aplicação foi containerizada para garantir que o ambiente de execução seja idêntico na máquina do desenvolvedor e no servidor de avaliação, eliminando o problema de "funciona na minha máquina".

-----

## 🚀 Como Executar

### Pré-requisitos

  * Docker instalado.

### Passo 1: Construir a Imagem

```bash
docker build -t tech-challenge-fase4 .
```

### Passo 2: Rodar o Container

```bash
docker run -p 8000:8000 tech-challenge-fase4
```

A API estará disponível em: **[http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)**

-----

## 🧪 Testando a API (Exemplo Real)

Você pode validar a API enviando uma requisição POST com os preços de fechamento dos últimos 60 dias.

**Exemplo via CURL:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{ "last_60_days": [30.5, 31.2, 31.0, ... (insira 60 valores)] }'
```

> **Dica:** Utilize o script `python gerar_teste.py` incluído no projeto para gerar um payload JSON atualizado com os dados reais da bolsa de hoje\!

-----

## 📊 Performance do Modelo

O modelo apresenta convergência estável sem sinais graves de *overfitting*, como demonstrado pelas curvas de Loss abaixo:

*(Insira aqui a imagem do gráfico de Loss ou Validação se desejar)*

**Resultados no Conjunto de Teste:**

  * **MAE:** R$ 0.68
  * **MAPE:** 2.25%

-----


## 👥 Autores do Projeto

| Membro | LinkedIn | GitHub |
|:--- |:---:|:---:|
| **Andrea Sakai** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/andr%C3%A9a-sakai-63751732/) | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/andreaono/) |
| **Bruno Ferreira** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bferreira-dev/) | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Bruno-Ferr) |
| **Glauberthy Cavalcanti** | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/glauberthy/) | [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/glauberthy) |
