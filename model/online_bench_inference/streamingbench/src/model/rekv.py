# ===== file: model/rekv.py =====
from operator import attrgetter


import torch
import os
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
from model.video_qa.rekv_offline_refactored import ReKVOfflineVQA
from model.llava_onevision_rekv import load_model





warnings.filterwarnings("ignore")
from src.model.modelclass import Model
class rekv(ReKVOfflineVQA):
    def __init__(self):
        self.sample_fps = 1
        self.retrieve_size = 64
        self.chunk_size = 1
        ReKV_Init(self)


    def Run(self, file, inp):
        return ReKV_Run(self, file, inp)
    def name(self):
        return "rekv"
    def initialize(self,  model_path=None):
        ReKV_Init(self) 
        
        
        
def ReKV_Init(self):
    self.qa_model, self.processor = load_model()

def ReKV_Run(self,file, inp):
    self.qa_model.past_memory_mean_token=[]
    video = self.load_video(file)
    if not isinstance(video, torch.Tensor):
        video_tensor = torch.from_numpy(video)
    else:
        video_tensor = video
    self.qa_model.clear_cache()
    self.qa_model.encode_init_prompt()
    self.qa_model.encode_video(video_tensor)
    response=self.qa_model.question_answering(inp)
    print("model_response:",response)
    return response

