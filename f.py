#!pip install diffusers transformers gradio accelerate
from flask import Flask, request, jsonify
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import io
from PIL import Image
from authtoken import auth_token


def generate_image_from_prompt(prompt):
    # Define the model ID
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"

    # Create the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
    pipe = pipe.to("cpu")

    # Generate the image
    image = pipe(prompt).images[0]
    
    return image

# Example usage:
prompt_text = """dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin,
extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans"""

generated_image = generate_image_from_prompt(prompt_text)

print(generated_image)


