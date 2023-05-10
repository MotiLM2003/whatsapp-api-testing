from flask import Flask, request, jsonify
import openai
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
openai.api_key = os.environ.get('OPENAI_API_KEY')


def generate_answer(question):
    model_engine = "text-davinci-002"
    prompt = (f"Q: {question}\nA:")

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7
    )

    answer = response.choices[0].text.strip()
    return answer


# answer = generate_answer('מה רמת החיים הממוצעת של בני האדם')
# with open('output.txt', 'w', encoding='utf-8') as file:
#     file.write(answer)



@app.route('/chatgpt', methods=['POST'])

def chatgpt():
    incoming_que = request.values.get('Body', '').lower()
    print(incoming_que)
    answer = answer = generate_answer(incoming_que)
    bot_resp = MessagingResponse()
    msg = bot_resp.message()
    msg.body(answer)
    return str(bot_resp)



if __name__ == '__main__':
    print('here')
    app.run(host='0.0.0.0', debug=True, port=5000)