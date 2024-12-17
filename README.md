# RAG for medical chatbot 🧠💬➕🏥

---

...Definition what's RAG, structure + usecases

## Installation Instruction🚀
---

### 1. Set up directory
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

5. (optional) Download the LLama2-Instruct-q4 model for this project from following link and put it in the directory \model :
https://huggingface.co/shrestha-prabin/llama-2-7b-chat.ggmlv3.q4_0/tree/main 


### 2. Set Up Your Own Vector Database with Pinecone

1. Create a Pinecone account by visiting [Pinecone's official website](https://www.pinecone.io/).

2. After signing up, go to your Pinecone dashboard, retrieve your API key, and save it securely.

3. Assign your API Key into the project

Create the .env File
(Assuming you’re using WSL or Ubuntu) Run the following command:
```bash
nano .env
```
In the .env file, add your Pinecone API key like this:
```ini
PINECONE_API_KEY= "XXXXXXXXXXXXXXXXXXXX"
``` 
Replace XXXXXXXXXXXXXXXXXXXX with your actual Pinecone API key.
Save (Ctrl+S) and Exit (Ctrl+X) the file

4. Create your own Vector database


...

```bash
python3 store_index.py
```

...

### 3. Download LLama Llama-3.2-3B-Instruct Model

The model will automatically be downloaded when you run the app.py script, which will be covered in the next section.
But before you can download the model:

1. Accept Meta LLaMA Terms and Conditions
    Go to the Meta [LLaMA Responsible Use Guide](https://www.llama.com/responsible-use-guide/).
    Review and accept the terms and conditions to gain access to the LLaMA model.

2. Generate a Hugging Face Access Token
    Log in or sign up for a [Hugging Face](https://huggingface.co/) with the same registered email with LLamA Terms and Condition.
    Create a User Access Token:
        Click on your profile image (top-right corner).
        Select "Access Tokens" from the dropdown menu.
        Click on "Create new token".
        Choose "Read" as the token type.
        Give the token a name (e.g., "LLaMA-access").
        Copy the generated token.

    Back to cmd window, run the following command to authenticate with Hugging Face:
    ```bash
    huggingface-cli
    ```
    When prompted, paste the token you copied earlier.
    Now you should have access to download the model


## Run the programm 🚀
---

```bash
python3 app.py
```


