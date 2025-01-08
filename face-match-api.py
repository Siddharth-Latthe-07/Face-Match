from fastapi import FastAPI, File, UploadFile, HTTPException
import face_recognition
import pickle
import os

app = FastAPI()

DATASET_PATH = "/home/siddharth/Desktop/face_recg/dataset"  # Update this to your dataset path
ENCODINGS_FILE = "encodings.pkl"

# Utility function to update encodings
def update_encodings(dataset_path, encodings_file):
    # Load existing encodings if the file exists
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as file:
            data = pickle.load(file)
        existing_encodings = data["encodings"]
        existing_images = set(data["images"])  # Use a set for faster lookups
    else:
        existing_encodings = []
        existing_images = set()

    # Process new images
    new_encodings = []
    new_images = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename not in existing_images:
            image_path = os.path.join(dataset_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                new_encodings.append(face_encoding)
                new_images.append(filename)
            else:
                print(f"No face found in {filename}")

    # Append new data to the existing data
    updated_encodings = existing_encodings + new_encodings
    updated_images = list(existing_images) + new_images

    # Save updated encodings back to the file
    with open(encodings_file, "wb") as file:
        pickle.dump({"encodings": updated_encodings, "images": updated_images}, file)

    return len(new_images), len(existing_images)

# Endpoint to generate/update encodings
@app.post("/generate-encodings")
async def generate_or_update_encodings_endpoint():
    new_count, existing_count = update_encodings(DATASET_PATH, ENCODINGS_FILE)
    return {
        "message": "Encodings updated successfully!",
        "new_images_processed": new_count,
        "total_images_in_dataset": new_count + existing_count
    }

# Endpoint to upload and match
@app.post("/upload-and-match")
async def upload_and_match(file: UploadFile = File(...)):
    if not os.path.exists(ENCODINGS_FILE):
        raise HTTPException(status_code=400, detail="Encodings file not found. Generate encodings first.")

    # Load precomputed encodings
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    encodings = data["encodings"]
    image_names = data["images"]

    # Load user image
    image_data = await file.read()
    with open("temp_uploaded_image.jpg", "wb") as temp_file:
        temp_file.write(image_data)
    user_image = face_recognition.load_image_file("temp_uploaded_image.jpg")

    user_face_locations = face_recognition.face_locations(user_image)
    if len(user_face_locations) == 0:
        os.remove("temp_uploaded_image.jpg")
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    user_face_encoding = face_recognition.face_encodings(user_image, user_face_locations)[0]

    # Compare with dataset encodings
    distances = face_recognition.face_distance(encodings, user_face_encoding)
    similarity_percentages = (1 - distances) * 100

    os.remove("temp_uploaded_image.jpg")  # Clean up the temporary file

    # Get matches above the 50% threshold
    matches = []
    for i, similarity in enumerate(similarity_percentages):
        if similarity > 50:  # Match threshold set to 50%
            matches.append({
                "image_name": image_names[i],
                "similarity": f"{similarity:.2f}%"
            })

    if matches:
        return {
            "message": "Matches found!",
            "matches": matches
        }
    else:
        return {
            "message": "No good match found in the dataset.",
        }
