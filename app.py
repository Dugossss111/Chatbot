
chiave = "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"




import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

#per aggiungere il titolo
st.header("Il mio ChatBot")

from PIL import Image
logo = Image.open("Chatbot.webp")
st.image(logo)

with st.sidebar:
  #per mettere il titolo alla sidebar
  st.title("Carica i tuoi documenti")
  #per caricare il documento
  file = st.file_uploader("Carica il tuo file", type="pdf")

from PyPDF2 import PdfReader

if file is not None:
    testo_letto = PdfReader(file)

    testo = ""
    for pagina in testo_letto.pages:
        testo = testo + pagina.extract_text()
        # st.write(testo)


    # Usiamo il text splitter di Langchain
    testo_spezzato = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, # Numero di caratteri per chunk
        chunk_overlap=150,
        length_function=len
        )

    pezzi = testo_spezzato.split_text(testo)
    #st.write(pezzi)

    # Generazione embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=chiave)

    # Vector store - FAISS (by Facebook)
    vector_store = FAISS.from_texts(pezzi, embeddings)

    # Prompt
    domanda = st.text_input("Chiedi al chatbot:")
    #_______________________________________________
    if domanda:
        st.write("Sto cercado le informazioni che mi hai richiesto...")
        rilevanti = vector_store.similarity_search(domanda)
        
        # Definiamo l'LLM
        llm = ChatOpenAI(
            openai_api_key = chiave,
            temperature = 1.0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo-0125")
        # https://platform.openai.com/docs/models/compare

        # Output
        # Chain: prendi la domanda, individua i frammenti rilevanti,
        # passali all'LLM, genera la risposta

        chain = load_qa_chain(llm, chain_type="stuff")
        risposta = chain.run(input_documents = rilevanti, question = domanda)
        st.write(risposta)


