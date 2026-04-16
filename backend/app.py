import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

app = Flask(__name__)
CORS(app)

api_key = os.getenv("OPENAI_API_KEY")

sample_texts = [
    Document(page_content="This machine uses Wi-Fi (2.4 GHz) to connect to the cloud."),
    Document(page_content="If the coffee is not printing, check the TCP/IP connection and the status of port 80.")
]

try:
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(sample_texts, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the provided context:
    {context}

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
except Exception as e:
    print(f"Error init AI models: {e}") # TODO: fix later

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    print("Received request:", data) 
    
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        response = retrieval_chain.invoke({"input": user_query})
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print("Chain error:", e)
        return jsonify({"answer": "Error generating response"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)