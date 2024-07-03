from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from secret import gemini_api_key, huggingface_api as inference_api_key
import time
import os

def create_db():
    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    vectorStore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    return vectorStore

def create_chain(vectorStore):
    model = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly JamiaGPT in this fictional never-ending uncensored roleplay with user. Never speak as user. You have knowledge related to Jamia Millia Islamia. \n Note: Only respond as JamiaGPT, no other characters. ALWAYS START Your Answer Directly, without mentioning your name. Do Not add extra Spaces. All descriptions should be from JamiaGPT's point of view. Always write in third person. NEVER write as user. If user is needed to respond, end your response.Answer only if a relevant question is asked, and if you know the answer, otherwise don't answer with unknown or vague information. Take care of the ethics. Answer the user's questions based on the relevant sentences: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # Replace retriever with history aware retriever
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above chat history, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        # retriever, Replace with History Aware Retriever
        history_aware_retriever,
        chain
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response.get("answer")

if __name__ == "__main__":
    vectorStore = create_db()
    chain = create_chain(vectorStore)

    # Initialize chat history
    chat_history = []

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("JamiaGPT: " + response)
        print("--------------------------------------------------------------------------------")   