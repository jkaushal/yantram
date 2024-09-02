import uuid

from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from langchain.chains.summarize import load_summarize_chain
from pinecone import Pinecone, ServerlessSpec


# Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)

        # Generate a unique ID using uuid
        doc_id = str(uuid.uuid4())

        # Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "doc_id": doc_id, "type=": filename.type, "size": filename.size,
                      "unique_id": unique_id},
        ))

    return docs


# Create embeddings instance
def create_embeddings_load_data():
    # embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Function to push data to Vector Store - Pinecone here
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    # pinecone.init(
    #     api_key=pinecone_apikey,
    #     environment=pinecone_environment
    # )
    pc = Pinecone(api_key="ad1a57df-eb6d-41a8-a63b-4ad3268e9a5e")
    pc.from_documents(docs, embeddings, index_name=pinecone_index_name)
    print("done......2")


# Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    pc = Pinecone(api_key="ad1a57df-eb6d-41a8-a63b-4ad3268e9a5e")
    index_name = pinecone_index_name
    index = pc.describe_index(index_name)
    return index


# Function to help us get relavant documents from vector store - based on user input
def similar_docs(query, k, pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, unique_id):
    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_environment
    )

    index_name = pinecone_index_name
    index = pull_from_pinecone(pinecone_apikey, pinecone_environment, index_name, embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k), {"unique_id": unique_id})
    # print(similar_docs)
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    # llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary