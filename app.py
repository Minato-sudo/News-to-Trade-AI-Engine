# app.py
# This file is used to trick Hugging Face's "Gradio" free tier into running our FastAPI backend.
# Hugging Face looks for an 'app' object in app.py.

from api.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
