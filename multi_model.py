import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, SPPF,C3
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized



class YOLOP_mod(nn.Module):
    def __init__(self, num_classes_bosch=16, num_classes_bdd=13):
        super(YOLOP_mod, self).__init__()
        
        self.num_classes_bosch = num_classes_bosch
        self.num_classes_bdd = num_classes_bdd
        
        self.backbone_p1 =  nn.Sequential(
        Focus(3,32,3),  #0
        Conv(32,64,3,2),  #1
        C3(64,64,1),#2
        Conv(64,128,3,2), #3
        C3(128,128,3),)#4
        
        self.backbone_p2 = nn.Sequential(
        Conv(128,256,3,2),#5
        C3(256,256,3),)#6
         
        self.backbone_p3 = nn.Sequential(
        Conv(256,512,3,2),  #7
        SPP(512,512,(5,9,13)), #8
        C3(512,512,1,False), #9
        Conv(512,256,1,1 ),  #10
        )
        
        self.backbone_p3_upsample = Upsample(None,2,'nearest') #11
        

        self.backbone_concat_p3up_p2 = Concat(1)

        self.backbone_out_csp =  BottleneckCSP(512,256,1,False)
        self.backbone_out_conv = Conv(256,128,1,1)
        self.backbone_out_upsample = Upsample(None,2,'nearest')
        
        self.backbone_out_concat = Concat(1) 

                                
        
        
        
        self.det_p1 = C3(256, 128, 1, False)
        self.det_p2 = Conv(128,128,3,2)
        self.det_p3 = Concat(1)
        self.det_p4 = C3(256,256,1,False)
        self.det_p5 = Conv(*[256, 256, 3, 2])
        self.det_p6 = Concat(1)
        self.det_p7 = C3(*[512, 512, 1, False])
        self.det_out_bosch = Detect (self.num_classes_bosch, 
                               [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]],
                               [128, 256, 512] )
        #self.det_out_bdd = Detect (self.num_classes_bdd, 
        #                       [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]],
        #                       [128, 256, 512] )
    
        self.lane_seghead = nn.Sequential(
        Conv(*[256, 128, 3, 1]),
        Upsample(*[None, 2, 'nearest']),
        C3(*[128, 64, 1, False]),
         Conv(*[64, 32, 3, 1]),   
            Upsample(*[None, 2, 'nearest']),
            Conv(*[32, 16, 3, 1]),
        C3(*[16, 8, 1, False]),
        Upsample(*[None, 2, 'nearest']),
            Conv(*[8, 2, 3, 1])
        )


        self.drivarea_seghead = nn.Sequential(
         Conv(*[256, 128, 3, 1]), 
            Upsample( *[None, 2, 'nearest']),
             C3(*[128, 64, 1, False]),
             Conv(*[64, 32, 3, 1]),
            Upsample(*[None, 2, 'nearest']),
            Conv(*[32, 16, 3, 1]),
            C3(*[16, 8, 1, False]),
             Upsample(*[None, 2, 'nearest']),
            Conv(*[8, 2, 3, 1])
        )
        #self.component_prefixes = [  for self.]
        '''
        self.encoder_params = []
        self.drivable_seg_head_params =[]
        self.lane_seg_head_params =[]
        self.det_head_params =[]

        self.encoder_params.extend(self.backbone_p1.parameters())
        self.encoder_params.extend(self.backbone_p2.parameters())
        self.encoder_params.extend(self.backbone_p3.parameters())
        self.encoder_params.extend(self.backbone_p3_upsample.parameters())
        self.encoder_params.extend(self.concat_p3up_p2.parameters())
        self.encoder_params.extend(self.b_out_csp.parameters())
        self.encoder_params.extend(self.b_out_conv.parameters())
        self.encoder_params.extend(self.b_out_upsample.parameters())
        self.encoder_params.extend(self.backbone_out_concat.parameters())

        self.drivable_seg_head_params.extend(self.driv_area_seg_head.parameters())
        self.lane_seg_head_params.extend(self.lane_area_seg_head.parameters())
        self.det_head_params.extend(self.det_p1.parameters() )
        self.det_head_params.extend(self.det_p2.parameters())
        self.det_head_params.extend(self.det_p3.parameters())
        self.det_head_params.extend(self.det_p4.parameters())
        self.det_head_params.extend(self.det_p5.parameters())
        self.det_head_params.extend(self.det_p6.parameters())
        self.det_head_params.extend(self.det_p7.parameters())
        self.det_head_params.extend(self.det_out.parameters())
        '''
        self.component_names = set([ k.split('_')[0] for k,v in self.named_parameters()])
        
        self.bdd_names = [str(i) for i in range(self.num_classes_bdd)]
        self.bosch_names = [str(i) for i in range(self.num_classes_bosch)]


        
        Detector = self.det_out_bosch
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s), bdd=False)
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            
        '''
        Detector = self.det_out_bdd
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s), bdd=True )
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)

            for x1,x2 in zip(Detector.stride, self.stride):
                assert x1 == x2
            self.stride = Detector.stride
        
        self._initialize_biases()
        '''
        
        initialize_weights(self)

    '''[ -1, Focus, [3, 32, 3]],   #0
    [ -1, Conv, [32, 64, 3, 2]],    #1
    [ -1, BottleneckCSP, [64, 64, 1]],  #2
    [ -1, Conv, [64, 128, 3, 2]],   #3
    [ -1, BottleneckCSP, [128, 128, 3]],    #4
    [ -1, Conv, [128, 256, 3, 2]],  #5
    [ -1, BottleneckCSP, [256, 256, 3]],    #6
    [ -1, Conv, [256, 512, 3, 2]],  #7
    [ -1, SPP, [512, 512, [5, 9, 13]]],     #8
    [ -1, BottleneckCSP, [512, 512, 1, False]],     #9
    [ -1, Conv,[512, 256, 1, 1]],   #10
    [ -1, Upsample, [None, 2, 'nearest']],  #11
    [ [-1, 6], Concat, [1]],    #12
    [ -1, BottleneckCSP, [512, 256, 1, False]], #13
    [ -1, Conv, [256, 128, 1, 1]],  #14
    [ -1, Upsample, [None, 2, 'nearest']],  #15
    [ [-1,4], Concat, [1]],     #16         #Encoder


      [ -1, BottleneckCSP, [256, 128, 1, False]],     #17
    [ -1, Conv, [128, 128, 3, 2]],      #18
    [ [-1, 14], Concat, [1]],       #19
    [ -1, BottleneckCSP, [256, 256, 1, False]],     #20
    [ -1, Conv, [256, 256, 3, 2]],      #21
    [ [-1, 10], Concat, [1]],   #22
    [ -1, BottleneckCSP, [512, 512, 1, False]],     #23
    [ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24
          '''
    def forward(self, tensor, bdd=False):
        x_p1 = self.backbone_p1(tensor)
        x_p2 = self.backbone_p2(x_p1)
        x_p3 =  self.backbone_p3(x_p2)
        x_p3_up = self.backbone_p3_upsample(x_p3)
        
        cat_p3_p2 = self.backbone_concat_p3up_p2([x_p3_up, x_p2])
        
        out_csp = self.backbone_out_csp(cat_p3_p2)
        out_conv = self.backbone_out_conv(out_csp)
        out_upsample = self.backbone_out_upsample(out_conv)
        
        cat_out_upsample_p1 = self.backbone_out_concat([x_p1, out_upsample])
      
        out1 = self.det_p1(cat_out_upsample_p1) 
        out2 = self.det_p2(out1)  
        out3 = self.det_p3([out2, out_conv])  # = Concat(1)
        out4 = self.det_p4(out3) 
        out5 = self.det_p5(out4) 
        out6 = self.det_p6([out5, x_p3])
        out7 = self.det_p7(out6)

        if not bdd:
            #detections = self.det_out_bdd([out1, out4, out7])
            #else:
            detections = self.det_out_bosch([out1, out4, out7])
        else:
            detections = None
    
        
        lane_area_preds = self.lane_seghead(cat_out_upsample_p1)
        driv_area_preds = self.drivarea_seghead(cat_out_upsample_p1)

        #if not self.training and (not bdd):
        #    detections = detections[0]
        return detections, lane_area_preds, driv_area_preds
    
    
        
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.det_out_bosch #self.model[self.detector_index]  # Detect() module

        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes_bosch - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        '''
        m = self.det_out_bdd #self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes_bdd - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        '''
