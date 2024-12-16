# RAG for medical chatbot 🧠💬➕🏥

---

...Definition what's RAG, structure + usecases

## Installation🚀
---
create virtual environment

```bash
conda create -n mchatbot python=3.10 -y
```
```bash
conda activate mchatbot
```
```bash
pip install -r requirements.txt
```

Download the LLama2-Instruct-q4 model for this project from following link and put it in the directory \model :

https://huggingface.co/shrestha-prabin/llama-2-7b-chat.ggmlv3.q4_0/tree/main 

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