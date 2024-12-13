from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

# A list to store the conversation
messages = [
    {"sender": "bot", "text": "Hello, I’m your medical assistant. How can I help you today?", "avatar": "bot.png"}
]

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
#print(PINECONE_API_KEY)

embeddings=download_huggingface_embeddings()

#Initializing the Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Define the Pinecone index
index_name = "medical-chatbot"

docsearch = LangchainPinecone.from_existing_index(index_name, embeddings) #from existing works with langchain-picone

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt":PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens' : 512,
                          'temperature': 0.4})

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=docsearch.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)

@app.route("/", methods=['GET', 'POST'])
def index():
    user_message = request.args.get('message') #Get the message
    print(user_message)
    
    if user_message:
        #Add user message to the list to display
        messages.append({"sender":"user", "text": user_message, "avatar":"user.png"})
        #Get the query for RAG to response
        result=qa_chain({"query": user_message})
        print("Response: ", result["result"])
        messages.append({"sender":"bot", "text": str(result["result"]), "avatar":"bot.png"})
    return render_template('chat.html', messages=messages)

'''
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg=request.form["msg"]
    input =msg
    print(input)
    result=qa_chain({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])
'''

if __name__=='__main__':
    app.run(debug=True)