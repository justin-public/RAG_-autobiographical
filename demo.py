import faiss
import re
from langchain_community.chat_models import ChatOllama
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

loader = PyPDFLoader("자서전 가상 100문100답.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

db = Chroma.from_documents(texts, embeddings)

llm = ChatOllama(model="ggml-model-Q5_K_M/Modelfile:latest")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
)

def ask_question(question):
    return qa_chain.run(question)

question = "PDF내용을 기반으로 자서전 생성과 관련된 6가지 주제를 만들어주세요?"
context = ask_question(question)
print(f"질문: {question}")
print(f"답변: {context}")