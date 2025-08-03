import cv2

print("Mendeteksi kamera...")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera ditemukan di index: {i}")
        cap.release()
