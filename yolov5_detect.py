import torch
import sys
import os
import cv2


def load_yolov5(model_path='/home/xrh1/experiments/extra/bosch/yolov5/runs/train/exp46_best_full_yolov5n/weights/best.pt'):
    
    model_full = torch.hub.load('ultralytics/yolov5', 'yolov5n', classes=15, pretrained=True, force_reload=True)
    checkpoint = torch.load(model_path)['model']
    model_full.model.load_state_dict(checkpoint.state_dict(), strict=False)

    return model_full

def infer_yolov5(model, image):
        class_names = ['Car', 'Stop sign', 'Red light', 'Yellow light', 'Green light', 'Crosswalk sign', 'Crosswalk', 'Highway sign', 'Precedence right sign', 'Black sign', 'Park sign', 'Wrong way sign', 'Pedestrian', 'End track', 'Black traffic light']

        results = model([image])
        results.names = class_names
        pred_list = results.display(show=False, render=False, crop=True)
        for pred_dict in pred_list:
                x0,y0,x1,y1 = [int(coord.cpu().item()) for coord in  pred_dict['box'] ]
                category_id = int(pred_dict['cls'].cpu().item())
                conf = pred_dict['conf'].cpu().item()
            
                image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2, 2)

                cv2.putText(image, f'{class_names[category_id] } {round(conf,3)}', 
                        (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,12), 2)
      
        return image     
                
