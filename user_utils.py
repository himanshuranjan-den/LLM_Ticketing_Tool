import pinecone
from langchain_pinecone import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import joblib

# Function to cerate embeddings
def create_embedding_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings

# function to pull data from Langchain
def pull_from_pinecone(pinecone_api_key, pinecone_index_name, embeddings):
    pc = Pinecone(pinecone_api_key=pinecone_api_key, embedding=embeddings, index_name=pinecone_index_name)
    index = pc.from_existing_index(embedding=embeddings, index_name=pinecone_index_name)
    return index

# Function to get the documents which are simllar to the query using Pinecone index
def get_smilar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Function to get the answers in best suitable format
def get_answer(docs, user_inputs):
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_inputs)
    return response

def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pk1')
    result = Fitmodel.predict([query_result])
    return result[0]