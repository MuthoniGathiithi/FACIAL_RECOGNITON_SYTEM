from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import os


detector=MTCNN()

def load_and_prepare_image(image_path):
    if os.path.isfile(image_path):
        image=cv2.imread(image_path)
        if image is not None:
            image_RGB=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_RGB
        else:
            print("Error loading image:")
            return None
        
             
    else:
        print("Invalid image path:")
        return None
    
def crop_detected_faces(Image_RGB):   
   face_details=detector.detect_faces(Image_RGB)
   if face_details:
       face_list=[]
       for faces in face_details:
           x,y,width,height=faces['box']
           cropped_face=Image_RGB[y:y+height,x:x+width]
           face_list.append(cropped_face)
           return face_list  
       
   else:
       print("No faces detected")
       return None    
   
   
   
   
   
   
if __name__ == "__main__":
    image_path = '/home/muthoni-gathiithi/Downloads/cat.jpeg'
    
    # Test step 1: Load image
    rgb_image = load_and_prepare_image(image_path)
    
    if rgb_image is not None:
        print(f"Image loaded successfully! Size: {rgb_image.shape}")
        
        # Test step 2: Detect faces with debugging
        face_details = detector.detect_faces(rgb_image)
        print(f"Raw detection results: {len(face_details)} faces found")
        
        # Show confidence scores
        for i, face in enumerate(face_details):
            confidence = face['confidence']
            print(f"Face {i+1}: confidence = {confidence}")
        
        cropped_faces = crop_detected_faces(rgb_image)(rgb_image)
        if cropped_faces:
            print(f"Final result: {len(cropped_faces)} faces!")