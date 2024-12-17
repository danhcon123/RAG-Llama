from src.helper import load_pdf, text_splitter, download_huggingface_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
#print(PINECONE_API_KEY)

#extract the data
extracted_data = load_pdf("data/")
text_chunks = text_splitter(extracted_data)
embeddings=download_huggingface_embeddings()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Define the Pinecone index
index_name = "medical-chatbot"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension for 'all-MiniLM-L6-v2' embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust your region
    )

text_chunks = [t.page_content for t in text_chunks]  # Replace with your text chunk source
docsearch = LangchainPinecone.from_texts(
    texts=text_chunks,
    embedding=embeddings,
    index_name=index_name
) 