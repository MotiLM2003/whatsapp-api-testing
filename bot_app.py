from flask import Flask, request, jsonify
import openai
from twilio.rest import Client
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
load_dotenv()
embeddings = OpenAIEmbeddings()


app = Flask(__name__)
openai.api_key = os.environ.get('OPENAI_API_KEY')
docs = ""


def create_db():
   # Get the root path of the project
    root_path = os.path.dirname(os.path.abspath(__file__))
    # Specify the file name or file path relative to the root directory
    file_name = 'data.txt'
    file_path = os.path.join(root_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        file_text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.create_documents([file_text])

    db = FAISS.from_documents(docs, embeddings)
    return db


db = create_db()
print(db)


def generate_answer(question, k=4):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    print('start sim')
    docs = db.similarity_search(question, k=k)
    print('end sim')
    docs_page_content = " ".join([d.page_content for d in docs])
   # Template to use for the system message prompt
    template = """
        You are a client server assistent of a company called "HiTexi" you will answer only  
        based on the content of this context: {docs}
        Only use the factual information from the context to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed.
        you will ALWAYS answer in the HEBREW language 
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

   # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    print('start chain')
    response = chain.run(question=question, docs=docs_page_content)
    print('start and chain')
    # response = response.replace("\n", "")
    return response


@app.route('/chatgpt', methods=['POST'])
def chatgpt():

    question = request.form.get('Body', '').lower()
    question = request.form.get('question')
    answer = generate_answer(question)

    account_sid = 'AC421d5e348f1ff08a9464b19fb75ba61d'
    auth_token = '9e047e6a061e2e3612193f3687971abb'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_='whatsapp:+14155238886',
        body=answer,
        to='whatsapp:+972586552717'
    )

    return answer


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
