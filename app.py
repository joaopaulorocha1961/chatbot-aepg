
    Aqui tens o código completo e consolidado. Este ficheiro app.py inclui a detecção automática de PDFs, a lógica bilingue, a nova estrutura da LangChain (sem o erro de importação) e o rodapé dinâmico com o link para o site do agrupamento.

Ficheiro: app.py
Python
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuração da Página
st.set_page_config(page_title="Assistente Virtual AEPG", page_icon="🏫", layout="centered")

# 2. Seletor de Idioma na Barra Lateral
lang = st.sidebar.selectbox("Idioma / Language", ["Português", "English"])

if lang == "Português":
    t = "Assistente Virtual - AE Paulo da Gama"
    info_msg = "Documentos oficiais carregados com sucesso."
    input_label = "Como posso ajudar?"
    footer_text = "\n\n---\n*Para mais informações e notícias atualizadas, visite o site oficial do Agrupamento: [aepg.pt](https://aepg.pt/)*"
    not_found_msg = "Nenhum ficheiro PDF encontrado no servidor."
    api_error = "Erro: Configura a OPENAI_API_KEY nos Secrets do Streamlit."
else:
    t = "Virtual Assistant - AE Paulo da Gama"
    info_msg = "Official documents loaded successfully."
    input_label = "How can I help you?"
    footer_text = "\n\n---\n*For more information and up-to-date news, please visit the official school website: [aepg.pt](https://aepg.pt/)*"
    not_found_msg = "No PDF files found on the server."
    api_error = "Error: Please configure the OPENAI_API_KEY in Streamlit Secrets."

st.title(t)

# 3. Verificação de Segurança (API Key)
if "OPENAI_API_KEY" not in st.secrets:
    st.error(api_error)
    st.stop()

# 4. Carregamento e Processamento de Documentos
@st.cache_resource
def setup_knowledge_base():
    # Deteta automaticamente todos os PDFs na raiz do projeto
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
            
    if not all_docs:
        return None
        
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Erro na OpenAI: {e}")
        return None

retriever = setup_knowledge_base()

# 5. Interface de Chat
if retriever:
    st.sidebar.success(info_msg)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar histórico
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input do utilizador
    if prompt := st.chat_input(input_label):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
            
            # Template com instruções de rigor e bilingues
            template = """És um assistente virtual do Agrupamento de Escolas Paulo da Gama. 
            Utiliza estritamente o contexto abaixo para responder.
            Se a resposta não estiver no contexto, diz que não encontras essa informação nos documentos oficiais.
            Responde sempre em {language}.
            
            Contexto: {context}
            Pergunta: {question}
            """
            
            rag_prompt = ChatPromptTemplate.from_template(template)
            
            # Construção da Chain (Lógica do Bot)
            chain = (
                {"context": retriever, "question": RunnablePassthrough(), "language": lambda x: lang}
                | rag_prompt | llm | StrOutputParser()
            )
            
            response = chain.invoke(prompt)
            full_response = response + footer_text
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning(not_found_msg)
