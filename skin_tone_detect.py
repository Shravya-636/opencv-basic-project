import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe face detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define skin tone ranges in LAB color space
skin_tone_ranges = {
    "Fair": [(190, 140, 140), (255, 170, 170)],
    "Medium": [(140, 120, 120), (190, 150, 150)],
    "Dark": [(80, 100, 100), (140, 130, 130)]
}

# Suggested colors for each skin tone
best_worst_colors = {
    "Fair": {
        "Best": [(203, 192, 255), (230, 216, 173), (193, 182, 255), (140, 230, 240), (250, 206, 135)],  
        "Worst": [(0, 255, 255), (0, 128, 0), (0, 0, 128), (0, 69, 255), (139, 0, 0)]  
    },
    "Medium": {
        "Best": [(0, 165, 255), (71, 99, 255), (30, 105, 210), (0, 140, 255), (19, 69, 139)],  
        "Worst": [(128, 0, 128), (255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 255, 255)]  
    },
    "Dark": {
        "Best": [(0, 215, 255), (144, 238, 144), (0, 140, 255), (32, 165, 218), (19, 69, 139)],  
        "Worst": [(169, 169, 169), (255, 255, 255), (192, 192, 192), (114, 128, 250), (180, 130, 70)]  
    }
}


def classify_skin_tone(lab_color):
    #Classifies skin tone based on LAB color ranges.
    for tone, (lower, upper) in skin_tone_ranges.items():
        if all(lower[i] <= lab_color[i] <= upper[i] for i in range(3)):
            return tone
    return "Unknown"

# Open camera
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            x = int(face_landmarks.landmark[10].x * w)
            y = int(face_landmarks.landmark[10].y * h)
            
            # Ensure x, y are within valid bounds
            x = min(max(x, 0), w - 1)
            y = min(max(y, 0), h - 1)

            # Get the forehead pixel color
            skin_color_bgr = frame[y, x]
            skin_color_lab = cv2.cvtColor(np.uint8([[skin_color_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            skin_tone = classify_skin_tone(skin_color_lab)
            

            # Get best and worst colors
            best_colors = best_worst_colors.get(skin_tone, {}).get("Best", [])
            worst_colors = best_worst_colors.get(skin_tone, {}).get("Worst", [])

            # Display Skin Tone text
            cv2.putText(frame, f"Skin Tone: {skin_tone}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display Best & Worst Colors
            y_best, y_worst = 100, 170 
            x_offset = 20

            for color in best_colors:
                cv2.rectangle(frame, (x_offset, y_best), (x_offset + 40, y_best + 30), color, -1)
                x_offset += 50 
            cv2.putText(frame, "Best Colors", (20, y_best - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            x_offset = 20  
            for color in worst_colors:
                cv2.rectangle(frame, (x_offset, y_worst ), (x_offset + 40, y_worst + 30), color, -1)
                x_offset += 50  
            cv2.putText(frame, "Worst Colors", (20, y_worst -5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Skin Tone Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
