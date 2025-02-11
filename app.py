from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nomic import NomicEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_MEDICAL_BOT_API_KEY')
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY')
)

embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5"
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    max_tokens=500,
)

index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
