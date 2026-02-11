# 📄 Chat com Leitor de PDF (RAG)

### LangChain + LangGraph + Streamlit + OpenRouter

Aplicação web interativa que permite fazer perguntas sobre um arquivo PDF.  
O sistema utiliza **RAG (Retrieval-Augmented Generation)** para buscar trechos relevantes no documento e gerar respostas baseadas exclusivamente no conteúdo do PDF.

---

## 🚀 Tecnologias Utilizadas

- Python  
- Streamlit  
- LangChain  
- LangGraph  
- OpenRouter (compatível com OpenAI API)  
- FAISS (vector store)  
- PyPDF  
- Embeddings  

---

## 🧠 Como Funciona

1. O usuário faz upload de um PDF.  

2. O sistema:
- Extrai o texto do documento  
- Divide em chunks  
- Gera embeddings  
- Armazena em um índice vetorial (FAISS)  

3. Quando uma pergunta é feita:
- O sistema recupera os trechos mais relevantes  
- O modelo responde **apenas com base nesses trechos**  
- Se a resposta não estiver no documento, ele informa  

---

## 🔐 Configuração da API

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua_chave_do_openrouter
```

---

## 🛠 Instalação

1️⃣ Criar ambiente virtual
```bash
py -m venv venv
venv\Scripts\Activate
```

2️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

▶️ Executar
```bash
streamlit run app.py
```

---
## 📌 Funcionalidades

- Upload de PDF via interface

- Busca semântica por similaridade

- Respostas fundamentadas no documento

- Indicação de páginas

- Memória de conversa com LangGraph

- Interface estilo chat

---
## 👩‍💻 Autora

Projeto desenvolvido por **Lorrane Aló Cruz**, como parte do aprendizado em Inteligência Artificial e Engenharia de Prompt.
