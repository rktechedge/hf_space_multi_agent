# Task Simplifier — HuggingFace Space (Gradio Multi-Agent Demo)

This repository contains a Gradio-based multi-agent demo designed to run on HuggingFace Spaces.
Place the files in a Space (Gradio), set the `GOOGLE_API_KEY` in the Space secrets, and the app will run server-side.

Files:
- app.py         # Gradio app (multi-agent pipeline)
- requirements.txt
- README.md

How to deploy:
1. Create a new Space on HuggingFace: https://huggingface.co/spaces
2. Choose **Gradio** as the SDK
3. Upload these files or link the GitHub repo
4. In the Space settings, add a secret `GOOGLE_API_KEY` with your Gemini key
5. The Space will build and become available. Use the UI to run the pipeline.

Notes:
- The app runs the pipeline server-side; this demonstrates multi-agent orchestration in a deployed environment.
- For long-running or production use, add persistence, authentication, rate-limiting, and proper secrets management.
