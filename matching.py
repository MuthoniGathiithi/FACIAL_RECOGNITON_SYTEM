import numpy as np
import sqlite3
import pickle
from detection import load_and_prepare_image, multi_scale_detect, crop_detected_faces
from normalization import normalize_entire_list
from facial_extraction import extract_features_from_entire_list


def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two face embeddings"""
    
    # 1: Calculate dot product
    dot_product = np.dot(embedding1, embedding2)
    
    # 2: Calculate magnitude of embedding1
    norm1 = np.linalg.norm(embedding1)
    
    #  3: Calculate magnitude of embedding2
    norm2 = np.linalg.norm(embedding2)
    
    # Step 4: Calculate similarity (dot / (norm1 * norm2))
    similarity = dot_product / (norm1 * norm2)
    
    # Step 5: Return similarity
    return similarity

def find_best_match(probe_embedding, threshold=0.6):
    """Find the best match for a probe embedding from the database
    
    Args:
        probe_embedding: Face embedding to match (512,)
        threshold: Minimum similarity score to consider a match
        
    Returns:
        tuple: (student_name, similarity_score) or (None, highest_score)
    """
    best_match_name = None
    best_match_id = None
    highest_similarity = -1
    
    # Load students from database
    try:
        conn = sqlite3.connect("students.db")
        c = conn.cursor()
        c.execute("SELECT student_id, name, embedding FROM students")
        students = c.fetchall()
        conn.close()
        
        # Compare against each student
        for student_id, name, embedding_blob in students:
            known_embedding = pickle.loads(embedding_blob)
            similarity = calculate_similarity(known_embedding, probe_embedding)
            
            print(f"DEBUG: Comparing with {name}: similarity = {similarity:.4f}")  # Add this
        
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_name = name
                best_match_id = student_id
        
        # Check if best match passes threshold
        if highest_similarity >= threshold:
            return best_match_name, highest_similarity
        else:
            return None, highest_similarity
            
    except Exception as e:
        print(f"Database error: {e}")
        return None, 0
    
def match_all_faces(normalized_face_list, threshold=0.6):
    """Match all normalized faces against the database"""
    matches = []
    embeddings = extract_features_from_entire_list(normalized_face_list)
    
    for i, embedding in enumerate(embeddings):
        # Flatten embedding from (1, 512) to (512,)
        flat_embedding = embedding.flatten()
        
        name, score = find_best_match(flat_embedding, threshold)
        if name:
            matches.append((name, score))
            print(f"Face {i+1}: Matched with {name} (Similarity: {score:.4f})")
        else:
            matches.append((None, score))
            print(f"Face {i+1}: No match found (Highest Similarity: {score:.4f})")
    
    return matches

# -------------------- TEST CODE -------------------- #
if __name__ == "__main__":
    # Test matching with a classroom photo
    image_path = '/home/muthoni-gathiithi/Downloads/testmyself.jpg'
    
    rgb_image = load_and_prepare_image(image_path)
    if rgb_image is not None:
        # Detect and crop faces
        face_details = multi_scale_detect(rgb_image)
        face_list, landmarks_list = crop_detected_faces(face_details, rgb_image)
        
        # Normalize faces
        normalized_faces = normalize_entire_list(face_list, landmarks_list)
        
        # Match all faces
        results = match_all_faces(normalized_faces, threshold=0.6)
        
        print(f"\n=== Attendance Summary ===")
        print(f"Total faces detected: {len(results)}")
        matched = sum(1 for name, _ in results if name is not None)
        print(f"Matched students: {matched}")
        print(f"Unknown faces: {len(results) - matched}")