import streamlit as st
from dotenv import load_dotenv
from pages.admin_utils import *
import constants
import os

def main():
    # print('Inside main function')
    load_dotenv()
    st.set_page_config(page_icon=":robot:", page_title="Dump PDF to Pinecone - Vector store")
    st.title("Please upload your file here!!")

    # Uplaod the PDF file
    pdf = st.file_uploader("Only files allowed are PDF!!", type=['pdf'])

    # Extract the whole text from PDF file
    if pdf is not None:
        with st.spinner('Processing your request.....'):
            text = read_pdf_data(pdf=pdf)
            st.write(":book: Reading PDF done!!")
            
            # Split  data into chunk
            docs_chunks = split_data(text=text)
            st.write(":knife: Splitting data into chunk done!!")
            
            # Create embedding
            embeddings = create_embedding_load_data()
            st.write(":1234: Embedding done!!")

            # Push data into Pinecone
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            index = push_to_pinecone(index_name=constants.PINECONE_INDEX, api_key=pinecone_api_key, embedding=embeddings, docs=docs_chunks)
            st.write(":blue_book: Pushed data into Vector store")

        st.success("Successfully pushed data into Embedding")

if __name__ == '__main__':
    print('Got hold of main')
    main()

# main()