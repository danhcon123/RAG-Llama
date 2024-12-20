#Using only Retrieval to answer the user's question
prompt_template_1 = """
You are a compassionate neuropsychology assistant. 
Use the following context to answer the patient's question clearly and concisely.
If the answer is not in the context, say "I don't know." Only answer the question.

{context}

User's question: {question}  
Answer:
"""

#Combine Retrieval + Generation (RAG Hybrid Logic)
prompt_template_2 = """
You are a compassionate neuropsychology assistant. 
Use the following context to answer the patient's question clearly and concisely.  
If the context does not provide the answer, use your general medical knowledge to help.  
If you still don't know, say "I don't know."

{context}

User's question: {question}  
Answer:
"""

#Combine Chat history Retrieval + Generation (RAG Hybrid Logic)
prompt_template_3 = """
You are a compassionate neuropsychology assistant.
Use the conversation history and the provided context to answer the patient's question clearly and concisely.  
If you don't know, say "I don't know."

Context: {context}

User's question: {question}
Answer:
"""


def extract_answer(response_text):
    """
    Extracts the final answer from the model response after the term 'Answer:'.

    Parameters:
    - response_text (str): The full response from the model.

    Returns:
    - str: The extracted answer text.
    """
    # Find the position where 'Answer:' appears
    answer_start = response_text.find("Answer:")
    
    if answer_start != -1:  # 'Answer:' was found
        # Extract everything after 'Answer:'
        answer = response_text[answer_start + len("Answer:"):].strip()
        # Case 1: Remove "I don't know" only if before this answer, there is no sentence got generated 
        if ". I don't know" in answer and answer.find(".I don't know") > 20:
            answer = answer[:answer.find(". I don't know")].strip()
        # Case 2: Remove everything starting from '(Note'
        if "Note:" in answer:
            answer = answer[:answer.find("Note:")].strip()
        return answer
    else:
        # If 'Answer:' not found, return a fallback message
        return "Unable to extract the answer."


