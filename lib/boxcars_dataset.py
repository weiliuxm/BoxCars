# -*- coding: utf-8 -*-
from config import BOXCARS_DATASET,BOXCARS_ATLAS,BOXCARS_CLASSIFICATION_SPLITS
from utils import load_cache
import cv2
import numpy as np

#%%
class BoxCarsDataset(object):
    def __init__(self, load_atlas = False, load_split = None, use_estimated_3DBB = False, estimated_3DBB_path = None):
        self.dataset = load_cache(BOXCARS_DATASET)
        self.use_estimated_3DBB = use_estimated_3DBB
        
        self.atlas = None
        self.split = None
        self.split_name = None
        self.estimated_3DBB = None
        self.X = {}
        self.Y = {}
        for part in ("train", "validation", "test"):
            self.X[part] = None
            self.Y[part] = None # for labels as array of 0-1 flags
            
        if load_atlas:
            self.load_atlas()
        if load_split is not None:
            self.load_classification_split(load_split)
        if self.use_estimated_3DBB:
            self.estimated_3DBB = load_cache(estimated_3DBB_path)
        
    #%%
    def load_atlas(self):
        self.atlas = load_cache(BOXCARS_ATLAS)
    
    #%%
    def load_classification_split(self, split_name):
        self.split = load_cache(BOXCARS_CLASSIFICATION_SPLITS)[split_name]
        self.split_name = split_name
       
    #%%
    def get_image(self, vehicle_id, instance_id):
        """
        returns decoded image from atlas in RGB channel order
        """
        return cv2.cvtColor(cv2.imdecode(self.atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)
        
    #%%
    def get_vehicle_instance_data(self, vehicle_id, instance_id, original_image_coordinates=False):
        """
        original_image_coordinates: the 3DBB coordinates are in the original image space
                                    to convert them into cropped image space, it is necessary to subtract instance["3DBB_offset"]
                                    which is done if this parameter is False. 
        """
        vehicle = self.dataset["samples"][vehicle_id]
        instance = vehicle["instances"][instance_id]
        if not self.use_estimated_3DBB:
            bb3d = self.dataset["samples"][vehicle_id]["instances"][instance_id]["3DBB"]
        else:
            bb3d = self.estimated_3DBB[vehicle_id][instance_id]
            
        if not original_image_coordinates:
            bb3d = bb3d - instance["3DBB_offset"]

        return vehicle, instance, bb3d 
            
       
    #%%
    #'train': [(0, 10), (1, 30), (2, 77),...] 
    def initialize_data(self, part):
        assert self.split is not None, "load classification split first"
        assert part in self.X, "unknown part -- use: train, validation, test"
        assert self.X[part] is None, "part %s was already initialized"%part
        data = self.split[part]
        x, y = [], []
        for vehicle_id, label in data: #vehicle_id,对应samples中的第几辆车；label对应车型
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"]) #vehicle_id对应的训练图像数量
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])#x是一个list，每个元素为一个二元的tuple，
            #tuple对应vehicle的序号和instance在当前vehicle多视角图像中的序号。
            y.extend([label]*num_instances) #得到instances对应的labels向量y。y是一个list，各元素对应一个instance。
        self.X[part] = np.asarray(x,dtype=int)

        y = np.asarray(y,dtype=int)
        y_categorical = np.zeros((y.shape[0], self.get_number_of_classes()))   #维度为（样本的数量，子类的数量）。hard数据集总共107个子类。
        y_categorical[np.arange(y.shape[0]), y] = 1   #每个sample对应的标签位置设为1。
        self.Y[part] = y_categorical
        


    def get_number_of_classes(self):
        return len(self.split["types_mapping"])
        
    
    #probabilities是个array，列的维度应该等于num_of_classes，行的维度等于样本个数
    def evaluate(self, probabilities, part="test", top_k=1):
        samples = self.X[part]
        assert samples.shape[0] == probabilities.shape[0]   
        assert self.get_number_of_classes() == probabilities.shape[1]   
        part_data = self.split[part]
        probs_inds = {}
        for vehicle_id, _ in part_data:
            probs_inds[vehicle_id] = np.zeros(len(self.dataset["samples"][vehicle_id]["instances"]), dtype=int)
        ##给每个instance从0开始编号
        for i, (vehicle_id, instance_id) in enumerate(samples):
            probs_inds[vehicle_id][instance_id] = i
            
        get_hit = lambda probs, gt: int(gt in np.argsort(probs.flatten())[-top_k:])
        hits = []
        hits_tracks = []
        for vehicle_id, label in part_data:
            inds = probs_inds[vehicle_id]
            hits_tracks.append(get_hit(np.mean(probabilities[inds, :], axis=0), label))#np.mean一个sample对应的多个instances的平均输出值
            for ind in inds:
                hits.append(get_hit(probabilities[ind, :], label))
                
        return np.mean(hits), np.mean(hits_tracks)
        
