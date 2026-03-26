import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Chatbot AEPG", page_icon="🏫")

# --- TRADUÇÃO ---
lang = st.sidebar.selectbox("Idioma / Language", ["Português", "English"])
if lang == "Português":
    t, label, info = "Assistente AE Paulo da Gama", "Pergunte algo...", "Documentos carregados com sucesso!"
else:
    t, label, info = "AEPG Assistant", "Ask something...", "Documents loaded successfully!"

st.title(t)

# --- VERIFICAÇÃO DE CHAVE ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Erro: Configura a OPENAI_API_KEY nos Secrets do Streamlit.")
    st.stop()

# --- CARREGAMENTO DE DOCUMENTOS ---
# Importante: Os nomes devem ser iguais aos ficheiros no GitHub
pdf_files = [
    "Projeto Educativo do Agrupamento.pdf",
    "Código de Conduta dos trabalhadores AEPG.pdf",
    "Politica de utilização saudavel do digital.pdf",
    "Calendario 25_26 NOVO.pdf"
]

@st.cache_resource
def setup_bot():
    docs = []
    for f in pdf_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            docs.extend(loader.load())
    
    if not docs:
        return None
        
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

retriever = setup_bot()

if retriever:
    st.success(info)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input(label):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Nova estrutura (Chain) sem usar RetrievalQA
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"])
            
            template = """Responde à pergunta com base apenas no contexto fornecido. 
            Se não souberes, diz que não encontraste no regulamento.
            Responde sempre em {language}.
            
            Contexto: {context}
            Pergunta: {question}
            """
            
            rag_prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough(), "language": lambda x: lang}
                | rag_prompt
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("⚠️ Ficheiros PDF não encontrados. Verifica se os nomes no código coincidem com o GitHub.")
