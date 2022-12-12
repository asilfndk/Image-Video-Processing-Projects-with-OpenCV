import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video1.mp4")
#cap = cv2.VideoCapture("video2.mp4")
#cap = cv2.VideoCapture("video3.mp4")
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 1)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    print(results.multi_face_landmarks)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

        for id, lm in enumerate(faceLms.landmark):
            h, w, c = img.shape
            x, y = int(lm.x * w), int(lm.y * h)
            print(id, x, y)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 65), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(50)