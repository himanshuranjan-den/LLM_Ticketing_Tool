from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms.openai import OpenAI
# from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import Pinecone
import pandas as pd
from sklearn.model_selection import train_test_split

def read_pdf_data(pdf):
    pdf_page = PdfReader(pdf)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text

def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    docs_chunk = text_splitter.create_documents(docs)
    return docs_chunk

def create_embedding_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings

def push_to_pinecone(api_key, index_name, embedding, docs):
    pc = Pinecone(pinecone_api_key=api_key, embedding=embedding, index_name=index_name) #aapinecone_api_key=api_keyadassd, 
    index = pc.from_documents(documents=docs, embedding=embedding, index_name=index_name)
    return index

# below function are for the SVM model
def read_data(data):
    df = pd.read_csv(data, delimiter=',', header=None)
    return df

def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeddings

def create_embeddings(df, embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

def split_train_test_data(df_sample):
    sentence_train, sentence_test, label_train, label_test = train_test_split(list(df_sample[2]), list(df_sample[1]), test_size=0.25, random_state=0)
    return sentence_train, sentence_test, label_train, label_test



def get_score(svm_classifier, sentence_test, label_test):
    score = svm_classifier.score(sentence_test, label_test)
    return score