from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

# Replace with your OpenAI API key
OPENAI_API_KEY = 'sk-proj-UYze18OpZOc7ObSiRvWZT3BlbkFJxL82QhzIIs7TPyKQHDoT'
API_URL = 'https://api.openai.com/v1/images/generations'

@app.route('/image', methods=['POST'])
def generate_image():
    try:
        user_input = request.json.get('text')  # Get user input from the request
        prompt = user_input   # Modify the prompt based on user input

        data = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "standard",
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }

        # Make the API request
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        result = response.json()

        if response.status_code == 200:
            image_url = result['data'][0]['url']
            return jsonify({"image_url": image_url})
        else:
            return jsonify({"error": "Image generation failed"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
