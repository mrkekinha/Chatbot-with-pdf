# Chatbot com PDF utilizando RAG, Groq e Pinecone

Este projeto implementa um chatbot que interage com documentos PDF, combinando técnicas de Recuperação Aumentada por Geração (RAG) com modelos de linguagem avançados e armazenamento vetorial. O objetivo é permitir que os usuários façam perguntas sobre o conteúdo de um PDF e recebam respostas precisas baseadas no documento fornecido.

## Sumário

1. [Introdução](#introdução)
2. [Funcionalidades](#funcionalidades)
3. [Arquivos no Repositório](#arquivos-no-repositório)
4. [Bibliotecas Necessárias](#bibliotecas-necessárias)
5. [Configuração do Ambiente](#configuração-do-ambiente)
6. [Execução do Aplicativo](#execução-do-aplicativo)
7. [Detalhamento do Código](#detalhamento-do-código)
8. [Possíveis Problemas e Soluções](#possíveis-problemas-e-soluções)
9. [Conclusão](#conclusão)

## Introdução

Este projeto foi desenvolvido para facilitar a interação com documentos PDF por meio de um chatbot. Utilizando técnicas de Recuperação Aumentada por Geração (RAG), o sistema combina a recuperação de informações relevantes de um documento com a geração de respostas coerentes, proporcionando uma experiência interativa e informativa ao usuário.

## Funcionalidades

- **Carregamento de PDFs**: O sistema permite o upload de documentos PDF para análise.
- **Divisão de Texto**: O conteúdo do PDF é dividido em segmentos menores para facilitar a manipulação e a recuperação de informações.
- **Geração de Embeddings**: Utiliza modelos de embeddings para representar semanticamente os segmentos de texto.
- **Armazenamento Vetorial**: Os embeddings são armazenados em um banco de dados vetorial para recuperação eficiente.
- **Interação via Chat**: Os usuários podem fazer perguntas relacionadas ao conteúdo do PDF, e o sistema responde com base nas informações contidas no documento.

## Arquivos no Repositório

- `app.py`: Script principal que executa o aplicativo Streamlit.
- `pdf.py`: Módulo responsável pelo processamento de documentos PDF.
- `main.ipynb`: Notebook Jupyter com exemplos e testes do sistema.
- `requirements.txt`: Lista de bibliotecas Python necessárias para o projeto.
- `data/`: Diretório que contém os arquivos PDF utilizados.

## Bibliotecas Necessárias

As principais bibliotecas utilizadas no projeto são:

- `os`: Para manipulação de variáveis de ambiente e caminhos de arquivos.
- `streamlit`: Para criação da interface web interativa.
- `dotenv`: Para carregamento de variáveis de ambiente de arquivos `.env`.
- `PyPDFLoader`: Para carregamento e leitura de arquivos PDF.
- `CharacterTextSplitter`: Para divisão do texto em segmentos menores.
- `CohereEmbeddings`: Para geração de embeddings semânticos dos textos.
- `pinecone`: Para armazenamento e recuperação vetorial eficiente.
- `langchain`: Para construção e gerenciamento de cadeias de processamento de linguagem natural.
- `ChatGroq`: Para interação com modelos de linguagem avançados.

## Configuração do Ambiente

1. **Instalação das Bibliotecas**: Execute o comando abaixo para instalar todas as dependências necessárias:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configuração das Variáveis de Ambiente**: Crie um arquivo `.env` na raiz do projeto e adicione as seguintes variáveis com suas respectivas chaves de API:

   ```env
   COHERE_API_KEY=your_cohere_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

   Substitua `your_cohere_api_key`, `your_pinecone_api_key` e `your_groq_api_key` pelas suas chaves de API correspondentes.

## Execução do Aplicativo

Após configurar o ambiente, execute o aplicativo com o seguinte comando:

```bash
streamlit run app.py
```

A interface do chatbot será aberta em seu navegador padrão, permitindo a interação com o documento PDF carregado.

## Detalhamento do Código

### 1. Importação de Bibliotecas e Configuração Inicial

```python
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
```

Aqui, importamos as bibliotecas necessárias para o funcionamento do aplicativo, incluindo ferramentas para manipulação de PDFs, geração de embeddings, armazenamento vetorial e criação da interface web.

### 2. Carregamento das Variáveis de Ambiente

```python
load_dotenv(find_dotenv())

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

Utilizamos a biblioteca `dotenv` para carregar as variáveis de ambiente a partir do arquivo `.env`, garantindo que as chaves de API estejam disponíveis no ambiente de execução.

### 3. Configuração da Interface com Streamlit

```python
st.set_page_config(page_title="Chat com PDF", layout="wide")
st.title("📚 Chat com o PDF usando RAG + Groq + Pinecone")
```

Configuramos a interface do aplicativo com um título e um layout amplo para melhor visualização.

### 4. Função de Configuração (`setup`)

```python
@st.cache_resource
def setup():
    # 1. Carrega e divide o PDF
    loader = PyPDFLoader("C:\\Users\\Maria Raquel\\Chatbot-with-pdf\\data\\2210.03629v3.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator="\n")
    docs = text_splitter.split_documents(documents)

    # 2. Gera embeddings
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)

    # 3. Inicializa Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "rag-demo"

    # Cria o índice se não existir
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,  # para cohere v3
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    # 4. Envia documentos para o índice (se necessário)
    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embedding=embeddings,
        index=index,
        namespace="default"
    )

    # 5. Config 
    ::contentReference[oaicite:0]{index=0}
        