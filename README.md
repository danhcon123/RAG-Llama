# RAG for chatbot - Llama Neurology assistant 🧠💬➕🏥

<div align="center">
    <img src="./static/images/top.PNG" alt="RAG Architecture" />
</div>
# Introduction
---

I’ve been working on a small project: a Retrieval Augmented Generation (RAG)-based “Neurological Assistant” 🧠 that uses Meta’s [LLaMA 3.2 (3B)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) model as its text generator. To power it up with real knowledge, I’m using the “all-MiniLM-L6-v2” sentence transformer to create embeddings from some books in PDF form on the human brain, cognition, and behavior. All these embeddings get stashed in a [Pinecone](https://www.pinecone.io/) vector database 🌲, so the assistant can fetch the info it needs whenever it’s stuck on a tough question. On top of that, I’ve also rolled out a user interface 💻 to make the whole thing more user-friendly. Now, instead of just a command-line tool, you’ve got a neat, interactive way to chat and learn.

The idea is to:
🚀 Turn complex neuroscience books into friendly conversations, making exploration less intimidating and more fun.
🚀 Foster interactive, engaging learning that helps you absorb and recall information faster.
🚀 Quickly retrieve learned knowledge to deepen understanding and connect new ideas.
🚀 Transform passive reading into a two-way dialogue, encouraging curiosity and insight.


# RAG 
(Definition + Usecase)

<div align="center">
    <img src="./static/images/RAG.PNG" alt="RAG Architecture" title="RAG Architecture Diagram" />
</div>

# Project Implementing, Pros and cons

...Definition what's RAG, structure + usecases


## Installation
---

For full installation instructions, see [INSTALL.md](INSTALL.md).


