from langchain import PromptTemplate
from langchain.chains import RetrievalQA
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import OpenAI

import os
print("OK")

#Extract data from the PDF

def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

extracted_data = load_pdf("data/")
#print(extracted_data)

#Create Text Chunks
def text_splitter(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

#text_chunks = text_splitter(extracted_data)
#print("length of my chunks:", len(text_chunks)) #6970

#Convert to Vector Embedding
def download_huggingface_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings=download_huggingface_embeddings()

#query_result=embeddings.embed_query("Hello world")
#print("Length", len(query_result)) #dimension 384


# Set Pinecone API Key in Environment
os.environ["PINECONE_API_KEY"] = "pcsk_48fRps_QeRX6JHqLTuf3h6HCjF7YCzcE1y4BKVFAy7U4HoisCTEkMYS9weG6Yoc55kCkiC"
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"  # Replace with your environment name

# Initialize Pinecone Client
'''
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
print(f"Using Pinecone index: {index_name}")

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use LangChain's Pinecone integration
text_chunks = [t.page_content for t in text_chunks]  # Replace with your text chunk source
docsearch = LangchainPinecone.from_texts(
    texts=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("Pinecone VectorStore created successfully!")
'''
index_name = "medical-chatbot"

docsearch = LangchainPinecone.from_existing_index(index_name, embeddings) #from existing works with langchain-picone

#query = "What is Allergie?"

#docs=docsearch.similarity_search(query, k=3)

#print("Result",docs) #Top 3 answer

prompt_template="""
Act as a doctor, who try to answer the patient's question as easy to understand and as comfort warmth as possible.
Use the following pieces of in information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:{context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:

"""
print(prompt_template)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt":PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens' : 1024,
                          'temperature': 0.8})

#print("llm done")

print("llm done")
'''
qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_document=True,
    chain_type_kwargs=chain_type_kwargs
)'''

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=docsearch.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)

print("Retrieval done")

#while True:
user_input=input(f"Input Prompt:")
result=qa_chain({"query": user_input})
print("Response: ", result["result"])
