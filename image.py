#!pip install diffusers transformers gradio accelerate
from flask import Flask, request, jsonify
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import io
from PIL import Image


'''def generate_image_from_prompt(prompt):
    # Define the model ID
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"

    # Create the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
    pipe = pipe.to("cpu")

    # Generate the image
    image = pipe(prompt).images[0]
    
    return image



app = Flask(__name__)

@app.route('/image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        prompt_text = data.get('prompt')

        # Generate the image
        generated_image = generate_image_from_prompt(prompt_text)

        # Convert the image tensor to a PIL image
        pil_image = Image.fromarray(generated_image.cpu().numpy())

        # Save the image to a buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return jsonify({'success': True, 'image': img_buffer.getvalue().decode('latin1')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)'''

from flask import Flask, request, render_template, send_file,jsonify
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import io
import base64
app = Flask(__name__)

def generate_image_from_prompt(prompt):
    # Define the model ID
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"

    # Create the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
    #pipe = pipe.to("cpu")

    # Generate the image
    image = pipe(prompt).images[0]

    # Save the image to a buffer (PNG format)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    return img_buffer

'''@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        generated_image = generate_image_from_prompt(user_input) 
        return generated_image
    return render_template('./index.html')'''


@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method=='POST':
        data=request.get_json()
        
        if 'prompt' in data:
            text = data['prompt']
            result = generate_image_from_prompt(text)
             # Convert BytesIO to base64
            image_bytes = result.getvalue()#img_buffer contains binary image data, which cannot be directly serialized to JSON. To resolve this, you can return the base64-encoded image data instead of the raw bytes.
            #Convert the BytesIO object to a base64-encoded string. This allows you to represent the image data as a text string that can be easily embedded in your API response.
            #You can use the base64 module in Python to achieve this conversion.
            #nstead of directly returning the BytesIO object, encode it as base64 and return the base64 string in your API response.
            #In your frontend, you can decode the base64 string back to bytes and display or process the image as needed.
             #For example, in JavaScript, you can create an <img> element with the src attribute set to the base64 string.
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            return jsonify({'image_base64': base64_image})
        else:
            return jsonify({'error':'Prompt Is not Provide in the Request'})

if __name__ == '__main__':
    app.run(debug=True,port=8000)































