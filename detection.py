from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os

# Initialize SCRFD detector
app = FaceAnalysis(providers=['CPUExecutionProvider'])

def load_and_prepare_image(image_path):
    """Load image and convert to RGB"""
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_RGB
        else:
            print("Error loading image")
            return None
    else:
        print("Invalid image path")
        return None

def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0
    return interArea / unionArea

def remove_duplicates(faces, iou_threshold=0.5):
    """Remove duplicate detections using IoU threshold"""
    unique_faces = []
    for f in faces:
        keep = True
        for uf in unique_faces:
            if iou(f.bbox, uf.bbox) > iou_threshold:
                keep = False
                break
        if keep:
            unique_faces.append(f)
    return unique_faces

def multi_scale_detect(image_rgb, scales=[640, 1024, 1280], det_thresh=0.3):
    """Run detection at multiple scales and merge results"""
    all_faces = []
    for size in scales:
        app.prepare(ctx_id=0, det_size=(size, size))
        faces = app.get(image_rgb)
        for f in faces:
            if f.det_score >= det_thresh:
                all_faces.append(f)
    # Remove duplicate faces
    all_faces = remove_duplicates(all_faces, iou_threshold=0.5)
    return all_faces

def crop_detected_faces(face_details, Image_RGB, det_thresh=0.3):
    """Crop faces and return cropped images + landmarks"""
    landmarks_list = []
    face_list = []
    for face in face_details:
        if face.det_score < det_thresh:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        cropped_face = Image_RGB[y1:y2, x1:x2]
        face_list.append(cropped_face)
        landmarks_list.append(face.kps.tolist())
    return face_list, landmarks_list


if __name__ == "__main__":
    image_path = '/home/muthoni-gathiithi/Downloads/many.png'
    rgb_image = load_and_prepare_image(image_path)
    if rgb_image is not None:
        print(f"Image loaded successfully! Size: {rgb_image.shape}")

        # Multi-scale detection
        face_details = multi_scale_detect(rgb_image, scales=[640, 1024, 1280], det_thresh=0.3)
        print(f"Multi-scale results: {len(face_details)} faces found")

        # Print confidence scores
        for i, face in enumerate(face_details):
            print(f"Face {i+1}: confidence = {face.det_score:.4f}")

        # Crop faces
        face_list, landmarks_list = crop_detected_faces(face_details, rgb_image, det_thresh=0.3)
        print(f"Final result: {len(face_list)} faces cropped!")
