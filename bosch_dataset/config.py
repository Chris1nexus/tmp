from lib.config import cfg
import copy


def get_bosch_cfg():
    bosch_cfg = copy.deepcopy(cfg)
    bosch_cfg.defrost()
    bosch_cfg.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256




    # DATASET related params
    #bosch_cfg.DATASET.DATAROOT = '/home/xrh1/datasets/bosch_dataset/video_datasets_merged_reduced/images'       # the path of images folder
    #bosch_cfg.DATASET.LABEL_JSON_PATH = '/home/xrh1/datasets/bosch_dataset/video_datasets_merged_reduced/images_annotations.json'      # the path of det_annotations folder
    bosch_cfg.DATASET.ROOT = '/home/xrh1/datasets/bosch_dataset/video_datasets_separated_reduced'
    bosch_cfg.DATASET.TRAIN_IMAGES = 'bfmc2020_online_1'       # the path of images folder
    bosch_cfg.DATASET.TRAIN_LABELS = 'bfmc2020_online_1_annotations.json'      # the path of det_annotations folder
    bosch_cfg.DATASET.VAL_IMAGES = 'bfmc2020_online_3'       # the path of images folder
    bosch_cfg.DATASET.VAL_LABELS = 'bfmc2020_online_3_annotations.json'      # the path of det_annotations folder


    bosch_cfg.DATASET.MASKROOT = ''                # the path of da_seg_annotations folder
    bosch_cfg.DATASET.LANEROOT = ''               # the path of ll_seg_annotations folder
    bosch_cfg.DATASET.DATASET = 'BoschDataset'
    bosch_cfg.DATASET.TRAIN_SET = ''
    bosch_cfg.DATASET.TEST_SET = ''
    bosch_cfg.DATASET.DATA_FORMAT = 'jpg'
    bosch_cfg.DATASET.SELECT_DATA = False
    bosch_cfg.DATASET.ORG_IMG_SIZE = [1232, 1640]

    return bosch_cfg