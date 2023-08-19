import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Livia Ellen](https://liviaellen.com/portfolio)')


def process_text(query, VectorStore,openai_key):
    st.write(f"API Key Used: {openai_key}")
    os.environ["OPENAI_API_KEY"] = openai_key
    docs = VectorStore.similarity_search(query=query, k=3)

    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        print(cb)
    st.header("Answer:")
    st.write(response)
    st.write('--')
    st.header("OpenAI API Usage:")
    st.text(cb)

def main():
    st.header("Chat with PDF 💬")
    st.write("This app uses OpenAI's LLM model to answer questions about your PDF file. Upload your PDF file and ask questions about it. The app will return the answer from your PDF file.")
    st.header("1. Pass your OPEN AI API KEY here")
    openai_key=st.text_input("**OPEN AI API KEY**")
    st.write("You can get your OpenAI API key from [here](https://platform.openai.com/account/api-keys)")
    os.environ["OPENAI_API_KEY"] = openai_key
    # if openai_key=='':
    #     load_dotenv()



    # upload a PDF file

    st.header("2. Ask My Own PDF")
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')

    # st.write(pdf)
    if st.button('Upload PDF'):
        if pdf is not None:
            pdf_reader = PdfReader(pdf)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                )
            chunks = text_splitter.split_text(text=text)

            # # embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
            # st.write(chunks)

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                # st.write('Embeddings Loaded from the Disk')s
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)


            # embeddings = OpenAIEmbeddings()
            # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:",value="Tell me about the PDF")
            # st.write(query)

            if query:
                process_text(query, VectorStore,openai_key)


    st.header("or.. Try it with this The Alchaemist PDF book")
    if st.button('Ask The Alchemist Book Questions'):
        with open("The_Alchemist.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:",value="Tell me about Santiago life")
        # st.write(query)
        if query:
            process_text(query, VectorStore,openai_key)




if __name__ == '__main__':
    main()
