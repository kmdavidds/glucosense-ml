from ultralytics import YOLO
import torch

if __name__ == '__main__':

    device = torch.device("cuda:0")

    # Load a model
    model = YOLO("yolov8n.pt").to(device=device) # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="./data.yaml", epochs=10, amp=False)  # train the model
    model.val()
