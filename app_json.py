import json
from base64 import b64decode
from io import BytesIO
from PIL import Image, ImageFile
import gradio as gr
import os

# Allow loading large PNGs
ImageFile.LOAD_TRUNCATED_IMAGES = True
ERROR_IMAGE = Image.new("RGB", (400, 200), color="red")

JSON_PATH = "gemini_response.json"  # ← your existing file

def load_image():
    if not os.path.exists(JSON_PATH):
        return "❌ gemini_response.json not found."

    try:
        # Load JSON
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        parts = data["candidates"][0]["content"]["parts"]

        img_b64 = None

        # ✅ Loop through all parts to find the image
        for part in parts:
            if "inline_data" in part:
                if part["inline_data"].get("mime_type") == "image/png":
                    img_b64 = part["inline_data"]["data"]
                    break  # Stop once we find the correct image part

        if not img_b64:
            print("⚠️ No PNG inline_data found in response. Returning placeholder.")
            return ERROR_IMAGE

        # ✅ Clean + decode Base64 safely
        img_b64 = img_b64.strip()
        img_bytes = b64decode(img_b64, validate=False)

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        print("✅ Successfully decoded image from response.")
        return img

    except Exception as e:
        print(f"❌ Error decoding image: {str(e)}")
        return ERROR_IMAGE


# Gradio UI
with gr.Blocks() as json_ui:
    gr.Markdown("## Gemini Image Viewer")

    gr.Interface(
        fn=load_image,
        inputs=None,     # No upload
        outputs=gr.Image(type="pil", label="Extracted Image"),
        title="Gemini Image Viewer",
        description="Automatically displays the image stored in gemini_response.json in this folder."
    ).queue()

# if __name__ == "__main__":
#     json_app.launch()
