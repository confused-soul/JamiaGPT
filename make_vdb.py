from secret import huggingface_api
inference_api_key = huggingface_api

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("JMI.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(separator="." ,chunk_size=400, chunk_overlap=150, length_function=len)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)

db.save_local("faiss_index")