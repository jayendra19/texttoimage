import tkinter as tk
from authtoken import auth_token
from diffusers import StableDiffusionPipeline
from torch import autocast
# Create the main app window
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Bud")

# Create the CTkEntry widget for user input
prompt = tk.Entry(app, width=50)  # Use a standard Entry widget
prompt.place(x=10, y=10)

# Create the StableDiffusionPipeline (model)
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, use_auth_token=auth_token)
pipe.to(device)

# Function to generate and display the image
def generate():
    user_input = prompt.get()
    with autocast(device):
        image = pipe(user_input, guidance_scale=8.5)["sample"][0]
    image.save('generatedimage.png')
    print("Image generated and saved as 'generatedimage.png'.")

# Create the "Generate" button
trigger = tk.Button(app, text="Generate", command=generate)
trigger.place(x=206, y=60)

# Run the app
app.mainloop()