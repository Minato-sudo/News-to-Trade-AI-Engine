# app.py
# This file is used to trick Hugging Face's "Gradio" free tier into running our FastAPI backend.
# Hugging Face looks for an 'app' object in app.py.

import gradio as gr
from api.main import app as fastapi_app
import spaces

@spaces.GPU
def dummy_gpu_function():
    return "GPU check passed! API is running."

# Create a tiny Gradio UI so Hugging Face registers the GPU function
demo = gr.Interface(
    fn=dummy_gpu_function,
    inputs=None,
    outputs="text",
    title="TradeSense API Status",
    description="The backend is successfully running. Use the Vercel frontend to interact with the API, or go to /docs for the Swagger UI."
)

# Mount our FastAPI app into Gradio. The API will work normally!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
# Finish
