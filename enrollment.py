from detection import load_and_prepare_image, multi_scale_detect, crop_detected_faces
from normalization import normalize_entire_list
from facial_extraction import extract_features_from_entire_list
import sqlite3
import pickle
import numpy as np

def enroll_student(student_id, student_name, image_paths):
    """Enroll a student by processing multiple photos and storing average embedding"""
    embeddings = []
    
    for path in image_paths:
        # Load image
        rgb_image = load_and_prepare_image(path)
        if rgb_image is None:
            print(f"Warning: Could not load {path}")
            continue
        
        # Detect faces
        face_details = multi_scale_detect(rgb_image)
        
        # Check if exactly one face detected
        if len(face_details) == 0:
            print(f"Warning: No face detected in {path}. Skipping.")
            continue
        elif len(face_details) > 1:
            print(f"Warning: Multiple faces ({len(face_details)}) detected in {path}. Skipping.")
            continue
        
        # Process the single face
        face_list, landmarks_list = crop_detected_faces(face_details, rgb_image)
        normalized_faces = normalize_entire_list(face_list, landmarks_list)
        face_embeddings = extract_features_from_entire_list(normalized_faces)
        
        # Flatten embedding from (1, 512) to (512,)
        if len(face_embeddings) > 0:
            embedding = face_embeddings[0].flatten()
            embeddings.append(embedding)
    
    # Check if we got any valid embeddings
    if len(embeddings) == 0:
        print(f"Enrollment failed for {student_name}: No valid embeddings extracted.")
        return False
    
    # Average all embeddings for robustness
    avg_embedding = np.mean(embeddings, axis=0)
    
    # Save to database
    try:
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        
        # Create table if doesn't exist
        c.execute("""CREATE TABLE IF NOT EXISTS students
                     (student_id TEXT PRIMARY KEY, 
                      name TEXT NOT NULL, 
                      embedding BLOB NOT NULL)""")
        
        # Insert or replace student
        c.execute("INSERT OR REPLACE INTO students (student_id, name, embedding) VALUES (?, ?, ?)",
                  (student_id, student_name, pickle.dumps(avg_embedding)))
        
        conn.commit()
        conn.close()
        
        print(f"{student_name} enrolled successfully with {len(embeddings)} photos!")
        return True
        
    except Exception as e:
        print(f"Database error: {e}")
        return False

# Test enrollment
if __name__ == "__main__":
    # Example: Enroll a student with multiple photos
    student_photos = [
        "/home/muthoni-gathiithi/Pictures/Screenshots/Screenshot from 2025-10-02 21-49-17.png",
        "/home/muthoni-gathiithi/Pictures/Screenshots/Screenshot from 2025-10-02 22-43-01.png",
        "/home/muthoni-gathiithi/Pictures/Screenshots/Screenshot from 2025-10-02 22-47-35.png"
    ]
    
    enroll_student("STU001", "Joyce Mwangi", student_photos)