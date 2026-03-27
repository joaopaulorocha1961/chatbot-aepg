import streamlit as st
import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Assistente Virtual AEPG", page_icon="🏫", layout="centered")

# --- CONFIGURAÇÃO DE IDIOMAS (TRILINGUE) ---
languages = {
    "Português (PT)": {
        "title": "Assistente Virtual - AEPG",
        "input": "Como posso ajudar?",
        "loading": "A ler documentos e site...",
        "footer": "\n\n---\n*Site oficial: [aepg.pt](https://aepg.pt/)*",
        "prompt_lang": "Portuguese",
        "rtl": False
    },
    "English (UK)": {
        "title": "AEPG Virtual Assistant",
        "input": "How can I help you?",
        "loading": "Reading documents and website...",
        "footer": "\n\n---\n*Official website: [aepg.pt](https://aepg.pt/)*",
        "prompt_lang": "English",
        "rtl": False
    },
    "Urdu (اردو)": {
        "title": "ورچوئل اسسٹنٹ - اے ای پالو دا گاما",
        "input": "میں آپ کی کیسے مدد کر سکتا ہوں؟",
        "loading": "...دستاویزات اور ویب سائٹ پڑھ رہا ہے",
        "footer": "\n\n---\n*سرکاری ویب سائٹ: [aepg.pt](https://aepg.pt/)*",
        "prompt_lang": "Urdu",
        "rtl": True
    }
}

if os.path.exists("LogoAEPG.png"): # Substitui pelo nome real do teu ficheiro
    st.sidebar.image("LogoAEPG.png", use_container_width=True)

# Seletor na barra lateral
selected_lang = st.sidebar.selectbox("Idioma / Language / زبان", list(languages.keys()))
lang_cfg = languages[selected_lang]

st.title(lang_cfg["title"])

# Ajuste de Direção para Urdu (RTL)
if lang_cfg["rtl"]:
    st.markdown("""<style> .stChatMessage { direction: rtl; text-align: right; } </style>""", unsafe_allow_html=True)

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Falta a OPENAI_API_KEY nos Secrets.")
    st.stop()

@st.cache_resource
def setup_knowledge_base():
    all_docs = []
    
    # --- 1. LEITURA DOS PDFs NO GITHUB ---
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    for f in pdf_files:
        try:
            loader = PyPDFLoader(f)
            all_docs.extend(loader.load())
        except: continue
    
    # --- 2. LEITURA REAL DO SITE ---
    # Podes adicionar mais páginas específicas (ex: /noticias) separadas por vírgula
    urls = ["https://aepg.pt/"] 
    try:
        web_loader = WebBaseLoader(urls)
        all_docs.extend(web_loader.load())
    except Exception as e:
        st.sidebar.warning(f"Erro ao ler o site: {e}")

    if not all_docs:
        return None
    
    # --- 3. DIVISÃO INTELIGENTE DO TEXTO ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    docs_chunks = text_splitter.split_documents(all_docs)
    
    try:
        embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
        
        # Criar base de dados em lotes para evitar erro 429
        vectorstore = FAISS.from_documents(docs_chunks[:30], embeddings)
        for i in range(30, len(docs_chunks), 30):
            vectorstore.add_documents(docs_chunks[i:i+30])
            time.sleep(1) 
            
        # k=10 permite que o bot "leia" muito mais contexto antes de responder
        return vectorstore.as_retriever(search_kwargs={"k": 10})
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

    if prompt := st.chat_input("Pergunte-me algo sobre o AEPG..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
            
            template = """És o assistente oficial do Agrupamento de Escolas Paulo da Gama.
            Utiliza o contexto fornecido (PDFs e Site) para responder de forma completa.
            Se a resposta não estiver clara, sugere consultar o site aepg.pt ou contactar a escola.
            Responde sempre em {language}.

            Contexto: {context}
            Pergunta: {question}
            """
            
            rag_prompt = ChatPromptTemplate.from_template(template)
            chain = ({"context": retriever, "question": RunnablePassthrough(), "language": lambda x: lang} 
                     | rag_prompt | llm | StrOutputParser())
            
            response = chain.invoke(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("A preparar o conhecimento... Verifique os ficheiros e a chave API.")
