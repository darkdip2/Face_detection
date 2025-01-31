import cv2
import os
import time
from tqdm import tqdm

OUTPUT_DIR=input("Enter your name...  ")
os.makedirs(OUTPUT_DIR,exist_ok=True)


cap=cv2.VideoCapture(0)


if not cap.isOpened():
    print('Could not access camera')
    exit()


print("press 'q' to stop recording")


frame_count=0
fps = 2 
frame_interval = 1 / fps
duration = 120  


with tqdm(total=duration, desc="Capturing frames", unit="s") as pbar:
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Frame", frame)

        elapsed_time = time.time() - start_time
        pbar.n = int(elapsed_time) 
        pbar.refresh()

        cv2.imwrite(f'{OUTPUT_DIR}/frame_{frame_count}.jpg',frame)
        frame_count+=1

        time.sleep(max(0, frame_interval - (time.time() - elapsed_time - start_time)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()