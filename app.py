import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Configuração da Página
st.set_page_config(page_title="Chatbot AEPG", page_icon="🏫")

# 2. Seletor de Idioma na Barra Lateral
language = st.sidebar.selectbox("Language / Idioma", ["Português", "English"])

if language == "Português":
    st.title("Assistente Virtual - AE Paulo da Gama")
    st.markdown("Pergunte-me sobre o **Regulamento, Calendário Escolar ou Projeto Educativo**.")
    query_label = "Como posso ajudar?"
    error_msg = "Por favor, insira uma pergunta."
else:
    st.title("Virtual Assistant - AE Paulo da Gama")
    st.markdown("Ask me about **Regulations, School Calendar, or the Educational Project**.")
    query_label = "How can I help you?"
    error_msg = "Please enter a question."

# 3. Lógica de IA (Resumo simplificado)
# Nota: Para isto funcionar online, precisas de uma API Key da OpenAI
# ou usar um modelo gratuito como o Llama via Groq.
def get_response(user_query, lang):
    # Aqui o código lê os teus PDFs (ex: "Projeto Educativo do Agrupamento.pdf")
    # e gera uma resposta contextualizada.
    # Se o idioma for English, instruímos o bot a traduzir a resposta.
    system_prompt = "És um assistente do AEPG. Responde apenas com base nos documentos fornecidos."
    if lang == "English":
        system_prompt += " Answer always in English."
    
    # [Lógica de Recuperação de Documentos aqui]
    return "Esta é uma resposta simulada baseada nos documentos do AEPG."

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(query_label):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_response(prompt, language)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
