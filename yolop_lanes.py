import os
import cv2
import torch
import argparse
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression

from lib.dataset.convert import id_dict
from bosch_dataset.config import get_bosch_cfg
from pycocotools.coco import COCO
import os


def load_yolop(model_path='/home/xrh1/Downloads/epoch-66.pth' ):
    bosch_cfg = get_bosch_cfg()
    labels_path = os.path.join(bosch_cfg.DATASET.ROOT,
                                  bosch_cfg.DATASET.TRAIN_LABELS)
    coco = COCO(labels_path)

    categories = coco.loadCats(coco.getCatIds())
    cat_names = [ cat['name'] for cat in sorted(categories, key=lambda cat_dict: cat_dict['id']) ]
    from multi_model import YOLOP_mod
    import torch
    model = YOLOP_mod(num_classes_bosch=15, num_classes_bdd=13)
        
    model.bosch_names = cat_names
    model.bdd_names =[ item[0] for item in  sorted(id_dict.items(), key=lambda x:x[1]) ] 


    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model = model.to('cuda')
    return model

def infer_yolop(model,  image):


    def resize_unscale(img, new_shape=(640, 640), color=114):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        canvas = np.zeros((new_shape[0], new_shape[1], 3))
        canvas.fill(color)
        # Scale ratio (new / old) new_shape(h,w)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
        new_unpad_w = new_unpad[0]
        new_unpad_h = new_unpad[1]
        pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

        dw = pad_w // 2  # divide padding into 2 sides
        dh = pad_h // 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

        canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

        return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


    height, width, _ = image.shape

    # convert to RGB
    img_rgb = image#img_bgr[:, :, ::-1].copy()

    # resize & normalize
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))

    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, 0)  # (1, 3,640,640)

    # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
    img = torch.tensor(img).to('cuda')
    det_out, da_seg_out, ll_seg_out = model(img)
    det_out = det_out[0]
    det_out = det_out.cpu()
    da_seg_out = da_seg_out.cpu()
    ll_seg_out = ll_seg_out.cpu()
    


    boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    if boxes.shape[0] == 0:
        #print("no bounding boxes detected.")
        #return
        pass
    else:
        # scale coords to original size.
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, 2] -= dw
        boxes[:, 3] -= dh
        boxes[:, :4] /= r

        #print(f"detect {boxes.shape[0]} bounding boxes.")

        img_det = img_rgb[:, :, ::-1].copy()
        
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2, conf, label = boxes[i]
            x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
            img_det = cv2.rectangle(img_det, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
            cv2.putText(img_det, f'{model.bosch_names[label] } {round(conf,3)}', 
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            

        

    # select da & ll segment area.
    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)


    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]

    # merge: resize to original size
    img_merge[color_mask != 0] = \
        img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        cv2.putText(img_merge, f'{model.bosch_names[label]} {round(conf,2):.2f}', 
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,12), 2)
    # da: resize to original size
    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.numpy().astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # ll: resize to original size
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.numpy().astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)



    return ll_seg_mask, img_merge