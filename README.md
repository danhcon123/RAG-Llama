# RAG for medical chatbot 🧠💬➕🏥

---

...Definition what's RAG, structure + usecases

## Installation instruction🚀
---

### Set up directory
1. Clone and Access the repository
```bash
git clone https://github.com/danhcon123/RAG-with-Llama2-as-medical-chatbot-.git
cd RAG-with-Llama2-as-medical-chatbot-
```

2. Create and Activate a Virtual Environment
```bash
python3.10 -m chatbot myenv
source chatbot/bin/activate
```
3. Follow the [PyTorch Installation Instructions](https://pytorch.org/get-started/locally/) to install the appropriate version for your system

4. Install all the required dependencies from the requirements.txt file:
```bash
pip install -r requirements.txt
```

5. (optional )Download the LLama2-Instruct-q4 model for this project from following link and put it in the directory \model :

https://huggingface.co/shrestha-prabin/llama-2-7b-chat.ggmlv3.q4_0/tree/main 

### Set up your own vector data base with Pinecone

Create your own Pinecone data base and API KEY under following links:

https://www.pinecone.io

Create a .env file in the root directory and add your Pinecone (vector data base) credentials as follows:

```ini
PINECONE_API_KEY= "XXXXXXXXXXXXXXXXXXXX"
``` 

Create your own Vector database

```bash
python3 store_index.py
```