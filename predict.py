from ultralytics import YOLO
import os

model = YOLO("./models/bestv1.pt")

image_folder = "./examples"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

results = model(image_files)

i = 0
for result in results:
    i+=1
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs  
    obb = result.obb  
    result.save(filename=f"./examples/{i}result.jpg")  