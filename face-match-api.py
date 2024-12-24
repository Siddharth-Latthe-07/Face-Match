from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import face_recognition
import pickle
import os
import numpy as np
from PIL import Image
OUTPUT_DIR = "./output"

app = FastAPI()

DATASET_PATH = "/home/siddharth/Desktop/face_recg/dataset"  # Update this to your dataset path
ENCODINGS_FILE = "encodings.pkl"

# Utility to generate encodings
def generate_encodings(dataset_path):
    encodings = []
    images = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dataset_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                encodings.append(face_encoding)
                images.append(filename)
            else:
                print(f"No face found in {filename}")
    return encodings, images

# Endpoint to generate encodings
@app.post("/generate-encodings")
async def generate_encodings_endpoint():
    encodings, images = generate_encodings(DATASET_PATH)
    with open(ENCODINGS_FILE, "wb") as file:
        pickle.dump({"encodings": encodings, "images": images}, file)
    return {"message": "Encodings generated and saved successfully!"}

# Combined endpoint: Upload image, find match, save output
@app.post("/upload-and-match")
async def upload_and_match(file: UploadFile = File(...)):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(ENCODINGS_FILE):
        raise HTTPException(status_code=400, detail="Encodings file not found. Generate encodings first.")

   
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    encodings = data["encodings"]
    image_names = data["images"]

    # Load user image
    image_data = await file.read()
    uploaded_image_path = os.path.join(OUTPUT_DIR, "uploaded_image.jpg")
    with open(uploaded_image_path, "wb") as temp_file:
        temp_file.write(image_data)
    user_image = face_recognition.load_image_file(uploaded_image_path)

    user_face_locations = face_recognition.face_locations(user_image)
    if len(user_face_locations) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    user_face_encoding = face_recognition.face_encodings(user_image, user_face_locations)[0]

    # Compare with dataset encodings
    distances = face_recognition.face_distance(encodings, user_face_encoding)
    best_match_index = np.argmin(distances)
    similarity_percentage = (1 - distances[best_match_index]) * 100

    if similarity_percentage > 50:  # Match threshold set to 50%
        matched_image_name = image_names[best_match_index]
        matched_image_path = os.path.join(DATASET_PATH, matched_image_name)

        # Save the matched image in the output directory
        matched_output_path = os.path.join(OUTPUT_DIR, "matched.jpg")
        matched_image = Image.open(matched_image_path)
        matched_image.save(matched_output_path)

        return {
            "message": "Match found!",
            "matched_image_name": matched_image_name,
            "similarity": f"{similarity_percentage:.2f}%",
            "output_image_path": matched_output_path
        }
    else:
        return {
            "message": "No good match found in the dataset.",
            "similarity": f"{similarity_percentage:.2f}%"
        }
