from ultralytics import YOLO
import torch

if __name__ == '__main__':

    device = torch.device("cuda:0")

    model = YOLO("yolov8n.pt").to(device=device)

    model.train(data="./data.yaml", epochs=10, amp=False) 
    model.val()
