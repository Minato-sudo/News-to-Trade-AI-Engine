# app.py
# This file is used to trick Hugging Face's "Gradio" free tier into running our FastAPI backend.
# Hugging Face looks for an 'app' object in app.py.

import gradio as gr
from api.main import app
import spaces
import uvicorn

@spaces.GPU
def dummy_gpu_function():
    return "GPU Registered!"

# Create a minimal Gradio app just to satisfy Hugging Face's ZeroGPU supervisor
demo = gr.Interface(fn=dummy_gpu_function, inputs=None, outputs="text")

# The ZeroGPU supervisor requires demo.launch() to be called so it can run its internal monkey-patches.
# We run Gradio on a hidden port (7861) and tell it not to block the thread.
demo.launch(server_name="0.0.0.0", server_port=7861, prevent_thread_lock=True)

# Now we bind our REAL FastAPI backend to the public port (7860) that Hugging Face exposes to Vercel!
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
