import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from pycocotools.coco import COCO
import os
from lib.dataset.convert import convert
from lib.utils import xyxy2xywh
from utils.augmentations import letterbox, augment_hsv, random_perspective, cutout


class BoschDataset:

    def __init__(self, cfg, 
                 is_train,
                 inputsize=640, 
                 transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        
        
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()

        self.data_root = cfg.DATASET.ROOT

        if is_train:
            self.images_dir = cfg.DATASET.TRAIN_IMAGES
            self.labels_file = cfg.DATASET.TRAIN_LABELS
        else:
            self.images_dir = cfg.DATASET.VAL_IMAGES
            self.labels_file = cfg.DATASET.VAL_LABELS
        
        self.img_root = Path(os.path.join(self.data_root, self.images_dir))
        self.label_root = Path(os.path.join(self.data_root, self.labels_file) )

        self.coco = COCO(self.label_root)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_dict = { cat['id']:cat for cat in self.categories}
        self.cat_names = [ item[1]['name'] for item in sorted(self.cat_dict.items(), key=lambda x: x[0])]

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
        
        self.db = self._get_db()
        
    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
      
        
        images_list = self.coco.loadImgs(self.coco.getImgIds())
        progressbar = tqdm(total=len(images_list))

        for img_data in images_list:
            imgid = img_data['id']
            
            annids = self.coco.getAnnIds(imgid)
            annotations = self.coco.loadAnns(annids)
            if len(annotations) == 0:
                continue
                
        


            image_path = os.path.join(self.img_root, img_data['file_name'])

            gt = np.zeros((len(annotations), 5))
            for idx, ann in enumerate(annotations):
                category_id = ann['category_id']
                x,y,w,h = ann['bbox']
                x1,y1 = x,y
                x2,y2 = x+w, y+h

                gt[idx][0] = category_id
                box = convert((width, height), (x1, x2, y1, y2))
                gt[idx][1:] = list(box)
                

            rec = [{
                'image': image_path,
                'label': gt
            }]
            progressbar.update(1)
            gt_db += rec
        print(f'database build finished: loaded {len(gt_db)} samples of {len(images_list)}')
        return gt_db

    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

        h, w = img.shape[:2]
        
        (img), ratio, pad = letterbox((img), resized_shape, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)
        
        det_label = data["label"]
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            
        if self.is_train:
            combination = (img,)
            img, labels = random_perspective(
                im=img,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )
            #print(labels.shape)
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, 
                        vgain=self.cfg.DATASET.HSV_V)
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)

                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
        
        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        #print(img.shape)
        img = np.ascontiguousarray(img)
        #print(img.shape)

        target = [labels_out, None, None]
        img = self.transform(img)

        return img, target, data["image"], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det= []
        
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
        
            
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)

        return torch.stack(img, 0), [torch.cat(label_det, 0)], paths, shapes

