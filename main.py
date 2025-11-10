# import uvicorn

# if __name__ == "__main__":
#     uvicorn.run(
#         "main:fast_app",     # points to your FastAPI app inside main_server.py
#         host="127.0.0.1",      # local-only (use 0.0.0.0 to allow LAN access)
#         port=3000,             # <-- change port if needed
#         reload=True            # auto restart on code change (dev mode)
#     )

import gradio as gr
import app as quitlens_page
import app_json as viewer_page
import os

with gr.Blocks(title="QuitLens") as app:
    gr.Markdown("# QuitLens")

    with gr.Tab("QuitLens Simulator"):
        quitlens_page.app_ui.render()

    with gr.Tab("Generated Image Viewer"):
        viewer_page.json_ui.render()

if __name__ == "__main__":
    app.launch(
        server_name="localhost",
        server_port=int(os.getenv("PORT", 3000)),
        show_error=True,
        share=True
    )