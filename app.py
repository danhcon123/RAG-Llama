from flask import Flask, render_template, jsonify, request, redirect,url_for
from src.helper import download_huggingface_embeddings
from src.prompt import *
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import HuggingFacePipeline
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()


# llama=3 to use llama 3.1-Instruct with 3B
# llama=2 to use llama 3.2-Instruct with 7B, which support by CTransformers for accellarating
llama=3

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
#prompt template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt":PROMPT}


#If you dont have GPU and want to run this on CPU, then get rid the bracket which comment this part and using this model
#and the CTransformers to run this model on CPU, this will be more ideal.
if llama == 2:
    llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens' : 512,#Input token from lama is 4096
                            'temperature': 0.4})
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=docsearch.as_retriever(search_kwargs={'k':1}),
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)
    
elif llama == 3:
    # ------------------------------
    # 2. Device Detection
    # ------------------------------
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = 0  # Typically the first GPU (NVIDIA)
        print("CUDA is available. Using GPU.")
    else:
        device = -1  # CPU
        print("CUDA not available. Using CPU.")
    # Model setup
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # Load tokenizer and model on GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            #device_map = 'cuda',
            torch_dtype=torch.float16 if cuda_available else torch.float32,
            low_cpu_mem_usage=True
        )
        # Create Hugging Face pipeline
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            #device=device,
            max_new_tokens=512,
            temperature=0.7
        )  
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={'k':2}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
  
    except Exception as e:
        print(f"Error loading model: {e}")
        # Handle the error or exit gracefully

    # Integrate into LangChain
    #llm = HuggingFacePipeline(pipeline=hf_pipeline)
                                       

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract JSON data from the request
        data = request.get_json()
        user_message = data.get('message')

        print("User Message:", user_message)

        # Append the user message to the messages list
        messages.append({"sender": "user", "text": user_message, "avatar": "user.png"})

        # Process with qa_chain to get the bot response
        result = qa_chain({"query": user_message})
        bot_response = result["result"]
        print("Bot Response:", bot_response)

        # Append the bot's response to the messages list
        messages.append({"sender": "bot", "text": bot_response, "avatar": "bot.png"})

        # Return bot's response as JSON
        return jsonify({"answer": bot_response})

    # For GET request, render the chat page with existing messages
    return render_template('chat.html', messages=messages)


if __name__=='__main__':
    app.run(debug=True)