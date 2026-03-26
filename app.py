import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Chatbot AE Paulo da Gama", page_icon="🏫")

# --- IDIOMA ---
lang = st.sidebar.selectbox("Idioma / Language", ["Português", "English"])
t = "Assistente AE Paulo da Gama" if lang == "Português" else "AEPG Assistant"
st.title(t)

# --- SECRETS ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Configura a OPENAI_API_KEY nos Secrets do Streamlit.")
    st.stop()

@st.cache_resource
def setup_bot():
    # Procura todos os ficheiros PDF na pasta principal
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return None
    
    all_docs = []
    for f in pdf_files:
        try:
            loader = PyPDFLoader(f)
            all_docs.extend(loader.load())
        except Exception as e:
            st.sidebar.warning(f"Erro ao ler {f}: {e}")
            
    if not all_docs:
        return None
        
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3}), pdf_files

# Inicialização
result = setup_bot()

if result:
    retriever, loaded_files = result
    st.sidebar.success(f"✅ {len(loaded_files)} documentos lidos.")
    with st.sidebar.expander("Ver ficheiros"):
        st.write(loaded_files)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Pergunte algo / Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
            
            template = """És um assistente virtual do Agrupamento de Escolas Paulo da Gama. 
            Responde com base no contexto. Se não souberes, diz que não encontras nos documentos.
            Responde sempre em {language}.
            
            Contexto: {context}
            Pergunta: {question}
            """
            rag_prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough(), "language": lambda x: lang}
                | rag_prompt | llm | StrOutputParser()
            )
            
            response = chain.invoke(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("⚠️ Nenhum ficheiro PDF encontrado no GitHub. Certifica-te de que os PDFs não estão dentro de pastas.")
