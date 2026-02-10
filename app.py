import os
import tempfile
from pathlib import Path
from typing import TypedDict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ========= ENV (.env) =========
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    api_key = api_key.strip()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ========= PROMPT (responde baseado no PDF) =========
SYSTEM_PROMPT = """
Você é uma assistente que responde perguntas APENAS com base no conteúdo do PDF fornecido.
Regras:
- Se a resposta não estiver no PDF, diga claramente: "Não encontrei isso no PDF."
- Cite trechos relevantes e, quando possível, cite a página (ex.: "p. 3").
- Seja direta, organizada e em português do Brasil.
"""


# ========= STATE DO LANGGRAPH =========
class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    messages: List[BaseMessage]
    answer: str


# ========= FUNÇÕES DE PDF/RAG =========
@st.cache_resource(show_spinner=False)
def build_embeddings():
    # Embeddings via OpenRouter (compatível com OpenAI API)
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


@st.cache_resource(show_spinner=False)
def build_llm():
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
    )


def load_pdf_to_docs(uploaded_file) -> List[Document]:
    # Salva o arquivo uploadado num temp e usa PyPDFLoader (preserva metadados como página)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()  # 1 Document por página, com metadata {"page": ...}

    # Limpa o arquivo temporário
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return docs


def build_vectorstore(docs: List[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    embeddings = build_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def format_context(docs: List[Document], max_chars: int = 6000) -> str:
    # Junta trechos com indicação de página
    parts = []
    total = 0
    for d in docs:
        page = d.metadata.get("page", None)
        page_str = f"p. {page + 1}" if isinstance(page, int) else "p. ?"
        text = d.page_content.strip()
        block = f"[{page_str}]\n{text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts)


# ========= LANGGRAPH (RAG) =========
def build_graph():
    llm = build_llm()

    def retrieve(state: RAGState) -> RAGState:
        vs: Optional[FAISS] = st.session_state.get("vectorstore")
        if vs is None:
            return {**state, "retrieved_docs": []}

        # Recupera top-k trechos mais relevantes
        docs = vs.similarity_search(state["question"], k=4)
        return {**state, "retrieved_docs": docs}

    def answer(state: RAGState) -> RAGState:
        if not state["retrieved_docs"]:
            answer_text = "Não encontrei isso no PDF. Você pode reformular a pergunta ou indicar a seção/página?"
            # mantém histórico
            messages = state["messages"] + [HumanMessage(content=state["question"])]
            messages = messages + [SystemMessage(content=answer_text)]
            return {**state, "messages": messages, "answer": answer_text}

        context = format_context(state["retrieved_docs"])

        prompt = [
            SystemMessage(content=SYSTEM_PROMPT.strip()),
            SystemMessage(content=f"CONTEÚDO RELEVANTE DO PDF:\n{context}"),
            HumanMessage(content=f"Pergunta: {state['question']}\n\nResponda usando apenas o conteúdo acima."),
        ]

        response = llm.invoke(prompt)
        answer_text = response.content

        # Atualiza histórico “resumido” pra não explodir tokens: salva só pergunta + resposta
        new_messages = state["messages"] + [
            HumanMessage(content=state["question"]),
            SystemMessage(content=answer_text),
        ]

        return {**state, "messages": new_messages, "answer": answer_text}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ========= STREAMLIT UI =========
st.set_page_config(page_title="📄 Pergunte ao PDF", page_icon="📄")
st.title("📄 Chat com leitor de PDF")

if not api_key:
    st.error("API key não encontrada. Confira o arquivo .env (OPENAI_API_KEY=...)")
    st.stop()

with st.sidebar:
    st.header("1) Envie seu PDF")
    uploaded = st.file_uploader("Upload do PDF", type=["pdf"])

    if st.button("Limpar conversa"):
        for k in ["thread_id", "graph", "ui_messages", "vectorstore", "pdf_name"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# thread_id para memória do LangGraph
if "thread_id" not in st.session_state:
    st.session_state.thread_id = os.urandom(8).hex()

# Compila o grafo 1x por sessão
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# UI messages
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Envie um PDF na barra lateral e depois me pergunte algo sobre ele 😊"}
    ]

# Se uploadou PDF, cria índice (vectorstore)
if uploaded is not None:
    if st.session_state.get("pdf_name") != uploaded.name:
        with st.spinner("Lendo PDF e criando índice de busca..."):
            docs = load_pdf_to_docs(uploaded)
            st.session_state.vectorstore = build_vectorstore(docs)
            st.session_state.pdf_name = uploaded.name
        st.success(f"PDF carregado e indexado: {uploaded.name}")

# Renderiza histórico
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Pergunte algo que está no PDF...")

if question:
    st.session_state.ui_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    if "vectorstore" not in st.session_state:
        st.session_state.ui_messages.append(
            {"role": "assistant", "content": "Antes, faça upload de um PDF na barra lateral 😊"}
        )
        with st.chat_message("assistant"):
            st.write("Antes, faça upload de um PDF na barra lateral 😊")
        st.stop()

    # chama LangGraph com memória
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # carrega histórico do grafo (mensagens resumidas)
    if "graph_messages" not in st.session_state:
        st.session_state.graph_messages = []

    result = st.session_state.graph.invoke(
        {
            "question": question,
            "retrieved_docs": [],
            "messages": st.session_state.graph_messages,
            "answer": "",
        },
        config=config,
    )

    answer_text = result["answer"]
    st.session_state.graph_messages = result["messages"]

    st.session_state.ui_messages.append({"role": "assistant", "content": answer_text})
    with st.chat_message("assistant"):
        st.write(answer_text)

    # (opcional) Mostrar fontes
    with st.expander("🔎 Ver trechos usados na resposta"):
        for i, d in enumerate(result["retrieved_docs"], start=1):
            page = d.metadata.get("page", None)
            page_str = f"p. {page + 1}" if isinstance(page, int) else "p. ?"
            st.markdown(f"**Trecho {i} — {page_str}**")
            st.write(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))
