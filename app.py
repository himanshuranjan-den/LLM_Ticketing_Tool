from dotenv import load_dotenv
import streamlit as st
from user_utils import *
import constants
import os
# Creating session variable

if "HR_tickets" not in st.session_state:
    st.session_state['HR_tickets'] = []
if "IT_tickets" not in st.session_state:
    st.session_state['IT_tickets'] = []
if "Transport_tickets" not in st.session_state:
    st.session_state['Transport_tickets'] = []
 


def main():
    load_dotenv()

    st.header("Automatic Ticket Classification Tool")

    # Capture user Input
    st.write("We are here to help you, please ask your question!!")
    user_input = st.text_input(":tropical_fish:")

    if user_input:

        # Creating embessing instance
        embeddings = create_embedding_load_data()

        # Function to pull index from pinecone
        pinecone_index = os.getenv('PINECONE_API_KEY') 
        index = pull_from_pinecone(pinecone_api_key = pinecone_index, pinecone_index_name = constants.PINECONE_INDEX, embeddings=embeddings)

        # This function will help us to fetch the relevant document from our vector store - Pincecone Index
        similar_docs = get_smilar_docs(index=index, query=user_input)

        # This will return fine tuned response by LLM
        response = get_answer(docs=similar_docs, user_inputs=user_input)
        st.write(response)

        button = st.button("Submit Ticket")
        if button:

            embeddings = create_embedding_load_data()
            query_result = embeddings.embed_query(user_input)

            department_value = predict(query_result=query_result)
            st.write("Your query has been submitted to {}".format(department_value))

            if department_value == "HR":
                st.session_state['HR_tickets'].append(user_input)
            elif department_value == "IT":
                st.session_state['IT_tickets'].append(user_input)
            else:
                st.session_state['Transport_tickets'].append(user_input)




if __name__ == "__main__":
    main()     

