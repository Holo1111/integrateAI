import os
from flask import Flask, request, jsonify, render_template
import openai
import logging
from flask_caching import Cache



app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Function to read the content of the file based on the company identifier
def read_text_file(company_id, file_name):
    """Read the contents of a text file with UTF-8 encoding."""
    file_path = f'static/companies/{company_id}/{file_name}'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise FileNotFoundError(f"File {file_path} not found")

# Function to get the answer from OpenAI's Chat model
@cache.memoize(timeout=300)  # Cache the result for 300 seconds (5 minutes)
def get_answer(document, question):
    try:
        prompt = f"You are an AI assistant. Given the following document and a question, provide a detailed answer based on the document.\n\nDocument:\n{document}\n\nQuestion:\n{question}\n\nAnswer:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message['content']
        return answer
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        company_id = data.get('company_id')
        question = data.get('question')
        
        if not company_id or not question:
            return jsonify({'error': 'Missing company_id or question'}), 400
        
        document = read_text_file(company_id, 'data_file.txt')
        answer = get_answer(document, question)
        return jsonify({'answer': answer})
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        return jsonify({'error': str(e)}), 404
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
