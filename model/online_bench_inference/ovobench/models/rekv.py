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


from utils.OVOBench import OVOBenchOffline



class Evalrekv(ReKVOfflineVQA,OVOBenchOffline):


    def __init__(self, args):
        self.args = args
        self.sample_fps = 1
        self.retrieve_size = getattr(args, 'retrieve_size', 64) if getattr(args, 'retrieve_size', None) is not None else 64
        self.chunk_size = 1
        self._model_init()
        #################################
        self.total_cuda_time =0
        self.max_mem=0
        ##################################
    def _model_init(self):
        self.qa_model, self.processor = load_model()



    def inference(self,file, inp):
        self.qa_model.past_memory_mean_token=[]
        video = self.load_video(file)
        video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2) 
        if not isinstance(video, torch.Tensor):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video_tensor
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        self.qa_model.encode_video(video_tensor)

        response=self.qa_model.question_answering(inp)
        response_lines = response.strip().splitlines()
        final_answer = response_lines[-1] if response_lines else ""
        print("model_final_answer:",final_answer)
        
        return final_answer

