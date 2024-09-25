from flask import Flask, request, jsonify, render_template
from final_v1 import create_c_agent, create_a_agent, create_b_agent

app = Flask(__name__)

a_agent = create_a_agent()
b_agent = create_b_agent()
chat_history = []


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    c_agent = create_c_agent(a_agent, b_agent, chat_history)
    response = c_agent.chat(user_input)
    
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Agent: {response.response}")
    
    return jsonify({'response': response.response})

if __name__ == '__main__':
    app.run(debug=True)