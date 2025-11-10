import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from io import BytesIO
from base64 import b64decode
import json
from PIL import ImageFile
import mediapipe as mp
import cv2

# ---------------------------------------------------------
# 1️⃣ Load environment variables
# ---------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("Gemini API Key loaded:", "✅" if GEMINI_API_KEY else "❌ Not Found")

# ---------------------------------------------------------
# 2️⃣ Configure Gemini model
# ---------------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
# models = genai.list_models()

# for m in models:
#     print(m.name)

model_name = "gemini-2.5-flash-image-preview"  # or "gemini-2.5-pro"
print(f"Using Gemini model: {model_name}")
model = genai.GenerativeModel(model_name)
ERROR_IMAGE = Image.new("RGB", (400, 200), color="red")

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Mouth + nose landmark indices
FULL_MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
INNER_LIPS_LANDMARKS = list(range(78, 88))  # lips contour region
MOUTH_LANDMARKS = FULL_MOUTH_LANDMARKS + INNER_LIPS_LANDMARKS
NOSE_BASE = 1  # nose bottom point

def extract_mouth_region(pil_img, return_overlay=False):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not res.multi_face_landmarks:
            print("⚠️ No face detected")
            return pil_img, pil_img if return_overlay else pil_img

        lm = res.multi_face_landmarks[0].landmark
        
        # nose reference (height anchor)
        nose_y = int(lm[NOSE_BASE].y * h)

        xs = [int(lm[i].x * w) for i in MOUTH_LANDMARKS]
        ys = [int(lm[i].y * h) for i in MOUTH_LANDMARKS]

        x_min, x_max = max(min(xs) - 40, 0), min(max(xs) + 40, w)
        y_min, y_max = max(min(ys) - 25, nose_y), min(max(ys) + 60, h)

        overlay = img.copy()
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0,255,0), 3)

        # Crop + upscale
        mouth = img[y_min:y_max, x_min:x_max]
        mouth = cv2.resize(mouth, (512, 512), interpolation=cv2.INTER_CUBIC)

        pil_crop = Image.fromarray(cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB))
        pil_overlay = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        return (pil_crop, pil_overlay) if return_overlay else pil_crop

# ---------------------------------------------------------
# 3️⃣ Define the simulation function
# ---------------------------------------------------------
def simulate_effect(image, scenario, age, gender, habits):
    try:
        # Convert NumPy array (Gradio format) → PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
            if image.mode != "RGB":
                image = image.convert("RGB")
        
        # Extract mouth only
        image, overlay = extract_mouth_region(image, return_overlay=True)


        # Dynamic prompt based on scenario
        if scenario == "Continue smoking":
            prompt = (
                "Enhance this mouth image to realistically show the long-term effects of tobacco: "
                "add yellow-brown stains on teeth, darker gums, and smoker’s palate discoloration."
            )
        else:
            prompt = (
                "Enhance this mouth image to show the healthy recovery after quitting smoking: "
                "whiter teeth, clean gums, bright lips, and normal oral texture."
            )

        try:
            # Generate content using Gemini
            # result = model.generate_content([prompt, image],  stream=False)
            result = model.generate_content(
            contents=[prompt, image],
            stream=False
            )
            response_json = result.to_dict()

            # Save to a file
            with open("gemini_response.json", "w", encoding="utf-8") as f:
                json.dump(response_json, f, indent=2)
            print("Gemini response received.")
            # Extract image bytes from the Gemini response
                    # ✅ Extract generated image from response
            print("Extracting generated image from Gemini response...")
            
            try:

                parts = response_json["candidates"][0]["content"]["parts"]

                img_b64 = None

                # ✅ Loop through all parts to find the image
                for part in parts:
                    if "inline_data" in part:
                        if part["inline_data"].get("mime_type") == "image/png":
                            img_b64 = part["inline_data"]["data"]
                            break  # Stop once we find the correct image part

                if not img_b64:
                    print("⚠️ No PNG inline_data found in response. Returning placeholder.")
                    return [overlay, ERROR_IMAGE]

                # ✅ Clean + decode Base64 safely
                img_b64 = img_b64.strip()
                img_bytes = b64decode(img_b64, validate=False)

                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                print("✅ Successfully decoded image from response.")
                return [overlay, img]


            except Exception as e:
                print("⚠️ Image extraction failed:", str(e))
                return [overlay, ERROR_IMAGE]
 

        except Exception as e:
            print(f"Gemini Error: {str(e)}")
            return [overlay, ERROR_IMAGE]
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return [ERROR_IMAGE, ERROR_IMAGE]

# ---------------------------------------------------------
# 4️⃣ Build the Gradio UI
# ---------------------------------------------------------
ui = gr.Interface(
    fn=simulate_effect,
    inputs=[
        gr.Image(label="Upload or Capture Mouth Image", sources=["upload", "webcam"]),
        gr.Radio(["Continue smoking", "Quit smoking"], label="Simulation Type"),
        gr.Textbox(label="Age (optional)"),
        gr.Dropdown(["Male", "Female", "Other"], label="Gender (optional)"),
        gr.Textbox(label="Smoking habits (duration/frequency)")
    ],
    outputs=[
        gr.Image(type="pil", label="Detected Mouth Region"),
        gr.Image(type="pil", label="Simulated Mouth Result")
    ],
    title="QuitLens: Oral Health Visualizer",
    description="Visualize oral health outcomes — continue vs quit smoking using Gemini 2.5 Flash Image model."
)


# Add post-processing hook
# ---------------------------------------------------------
# 5️⃣ Launch app
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        print("Launching QuitLens app...")
        ui.launch()
    except Exception as e:
        print(f"Error launching app: {str(e)}")
