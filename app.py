import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --- CONFIGURAÇÃO DA INTERFACE ---
st.set_page_config(page_title="Chatbot AEPG", page_icon="🏫")

# Seletor de Idioma
lang = st.sidebar.selectbox("Idioma / Language", ["Português", "English"])

if lang == "Português":
    title, info = "Assistente AE Paulo da Gama", "Pergunte sobre as regras e calendários."
    input_text = "Escreva aqui a sua pergunta..."
else:
    title, info = "AEPG Virtual Assistant", "Ask about rules and calendars."
    input_text = "Type your question here..."

st.title(title)
st.info(info)

# --- CARREGAMENTO DOS DOCUMENTOS ---
# Lista dos nomes exatos dos teus ficheiros no GitHub
docs_files = [
    "Calendario 25_26 NOVO.pdf",
    "Código de Conduta dos trabalhadores AEPG.pdf",
    "Politica de utilização saudavel do digital.pdf",
    "Projeto Educativo do Agrupamento.pdf"
]

@st.cache_resource
def load_data():
    documents = []
    for file in docs_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
    
    # IMPORTANTE: Precisas da chave da OpenAI para isto funcionar
    # Podes colocá-la nos "Secrets" do Streamlit Cloud
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = load_data()

if vectorstore is None:
    st.warning("⚠️ Erro: Chave API em falta ou ficheiros não encontrados.")
else:
    # --- INTERFACE DE CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(input_text):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Configura a IA para responder no idioma correto
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"]),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            
            context_prompt = f"Responda em {lang}. " + prompt
            response = qa.run(context_prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
