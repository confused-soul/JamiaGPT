from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

gemini_api_key = st.secrets['gemini_api_key']
inference_api_key = st.secrets['inference_api_key']

def create_db():
    embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    vectorStore = FAISS.load_local("faiss_index2", embedding, allow_dangerous_deserialization=True)
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

    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

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

vectorStore = create_db()
chain = create_chain(vectorStore)

st.set_page_config(page_title="JamiaGPT", page_icon="src/images/icon.jpg")

st.markdown('''
<style>
    .e1nzilvr1, .e1vs0wn31 {display:none}
    #jamiagpt {padding: 0}
</style>
''', unsafe_allow_html=True)

col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image(image="src/images/logo.png", width=60)
with col2:
    st.title(":green[JamiaGPT]")

disc = st.expander(label=":red[Disclaimer!]")
disc.write("This is still under development. The model is not perfect and may give irrelevant answers. Please be patient and ask questions related to Jamia Millia Islamia only.")

USER_AVATAR = "😁"
BOT_AVATAR = "🤖"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?", key="user_input", max_chars = 2000):
    # Display user message in chat message container
    st.chat_message("user", avatar=USER_AVATAR).markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = process_chat(chain, prompt, st.session_state.messages)
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
