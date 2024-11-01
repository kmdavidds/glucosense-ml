from ultralytics import YOLO
import os

# Load a model
model = YOLO("./models/bestv1.pt")

# Run batched inference on a list of images
# Get all filenames from folder ./examples
image_folder = "./examples"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

results = model(image_files)

i = 0
# Process results list
for result in results:
    i+=1
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=f"./examples/{i}result.jpg")  # save to disk