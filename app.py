import streamlit as st
import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Assistente Virtual AEPG", page_icon="🏫", layout="centered")

lang = st.sidebar.selectbox("Idioma / Language", ["Português", "English"])
t = "Assistente Virtual - AE Paulo da Gama" if lang == "Português" else "AEPG Assistant"
st.title(t)

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Configura a OPENAI_API_KEY nos Secrets.")
    st.stop()

@st.cache_resource
def setup_knowledge_base():
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if not pdf_files:
        return None
    
    all_docs = []
    for f in pdf_files:
        try:
            loader = PyPDFLoader(f)
            all_docs.extend(loader.load())
        except Exception:
            continue
    
    # --- SOLUÇÃO PARA O ERRO 429: DIVIDIR O TEXTO ---
    # Criamos pedaços de 1000 caracteres para não estourar o limite de tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_chunks = text_splitter.split_documents(all_docs)
    
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Criar a base de dados em pequenos lotes para evitar o erro de limite
        vectorstore = FAISS.from_documents(docs_chunks[:50], embeddings) # Começa com os primeiros 50
        
        # Adiciona o resto aos poucos (se houver muitos documentos)
        for i in range(50, len(docs_chunks), 50):
            vectorstore.add_documents(docs_chunks[i:i+50])
            time.sleep(1) # Pausa de 1 segundo para a OpenAI não bloquear
            
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Erro na OpenAI: {e}")
        return None

retriever = setup_knowledge_base()

if retriever:
    st.sidebar.success("Documentos carregados!" if lang == "Português" else "Documents loaded!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Pergunte algo..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
            
            footer = "\n\n---\n*Visite: [aepg.pt](https://aepg.pt/)*"
            template = "És um assistente do AEPG. Responde em {language} usando: {context}. Pergunta: {question}"
            
            rag_prompt = ChatPromptTemplate.from_template(template)
            chain = ({"context": retriever, "question": RunnablePassthrough(), "language": lambda x: lang} 
                     | rag_prompt | llm | StrOutputParser())
            
            response = chain.invoke(prompt)
            full_res = response + footer
            st.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})
else:
    st.warning("Nenhum PDF encontrado ou erro no processamento.")
