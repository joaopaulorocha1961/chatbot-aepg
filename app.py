import streamlit as st
import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# CORREÇÃO: Importação atualizada para a nova versão da biblioteca
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Assistente Virtual AEPG", page_icon="🏫", layout="centered")

# Configuração de Idioma
lang = st.sidebar.selectbox("Idioma / Language", ["Português", "English"])
if lang == "Português":
    t, input_l = "Assistente Virtual - AE Paulo da Gama", "Como posso ajudar?"
    footer_msg = "\n\n---\n*Visite o site oficial: [aepg.pt](https://aepg.pt/)*"
else:
    t, input_l = "AEPG Virtual Assistant", "How can I help you?"
    footer_msg = "\n\n---\n*Visit the official website: [aepg.pt](https://aepg.pt/)*"

st.title(t)

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Erro: Configura a OPENAI_API_KEY nos Secrets do Streamlit.")
    st.stop()

@st.cache_resource
def setup_knowledge_base():
    # Deteta PDFs na raiz do GitHub
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
    
    # Divide o texto em pedaços para evitar o erro 429 (Rate Limit)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_chunks = text_splitter.split_documents(all_docs)
    
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Cria a base de dados em pequenos lotes com pausas
        vectorstore = FAISS.from_documents(docs_chunks[:30], embeddings)
        
        for i in range(30, len(docs_chunks), 30):
            vectorstore.add_documents(docs_chunks[i:i+30])
            time.sleep(1) # Pausa técnica para a OpenAI não bloquear
            
        return vectorstore.as_retriever(search_kwargs={"k": 6})
    except Exception as e:
        st.error(f"Erro na OpenAI: {e}")
        return None

retriever = setup_knowledge_base()

if retriever:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input(input_l):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
            
            template = """És um assistente do AEPG. Responde em {language} usando o contexto: {context}. 
            Se não souberes, diz que não encontras nos documentos. Pergunta: {question}"""
            
            rag_prompt = ChatPromptTemplate.from_template(template)
            chain = ({"context": retriever, "question": RunnablePassthrough(), "language": lambda x: lang} 
                     | rag_prompt | llm | StrOutputParser())
            
            response = chain.invoke(prompt)
            final_answer = response + footer_msg
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
else:
    st.warning("⚠️ Carregue os PDFs no GitHub ou verifique a sua chave API.")
