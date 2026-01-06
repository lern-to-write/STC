import os
from .llava_qwen import LlavaQwenForCausalLM
from .stream_grounding_qwen import StreamGroundQwenForCausalLM
from ..multimodal_encoder.clip_encoder import CLIPVisionTower

# from ..multimodal_encoder.custom_clip_encoder import CustomCLIPVisionTower


# 

import torch
import torch.nn as nn

import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from typing import Optional, List


class QwenCompressor(nn.Module):
    def __init__(self, compressor):
        super().__init__()

        self.model_path = compressor
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.compressor = LlavaQwenForCausalLM.from_pretrained(self.model_path)
        self.select_layer = 100
        self.vision_encoder = CLIPVisionTower('YOUR_CLIP_CKPT_PATH')
    
    def forward_compress(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        qs_ids: Optional[torch.LongTensor]= None,
        qs_mask: Optional[torch.Tensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_large: Optional[torch.FloatTensor] = None,
        projector: Optional[torch.LongTensor] = None,
        select_layer: Optional[int] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        compress_tokens, clip_embeds, global_memory, loss, similarity = self.compressor.forward_token(
            input_ids,
            attention_mask,
            qs_ids,
            qs_mask,
            time_labels,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            images,
            images_large,
            projector,
            select_layer,
            return_dict,
            self.vision_encoder
        )
        return compress_tokens, clip_embeds, global_memory, loss, similarity
    
    def forward(self, clips, clips_large, seqs, compress_mask, qs, qs_mask, time_labels):
        return self.forward_compress(input_ids=seqs, attention_mask=compress_mask, qs_ids=qs, qs_mask=qs_mask, images=clips, images_large=clips_large, select_layer=self.select_layer, time_labels=time_labels)


class StreamQwenCompressor(nn.Module):
    def __init__(self, compressor):
        super().__init__()

        self.model_path = compressor
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.compressor = StreamGroundQwenForCausalLM.from_pretrained(self.model_path)
        self.select_layer = 100
    
    def forward_compress(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        qs_ids: Optional[torch.LongTensor]= None,
        qs_mask: Optional[torch.Tensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_large: Optional[torch.FloatTensor] = None,
        projector: Optional[torch.LongTensor] = None,
        select_layer: Optional[int] = None,
        insert_position: Optional[int] = None,
        ans_position: Optional[list] = None,
        silent_position: Optional[list] = None,
        ans_token: Optional[torch.LongTensor] = None,
        todo_token: Optional[torch.LongTensor] = None,
        silent_label: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        compress_tokens, clip_embeds, global_memory, loss, similarity = self.compressor.forward_token(
            input_ids,
            attention_mask,
            qs_ids,
            qs_mask,
            time_labels,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            images,
            projector,
            select_layer,
            insert_position,
            ans_position,
            silent_position,
            ans_token,
            todo_token,
            silent_label,
            return_dict,
        )
        return compress_tokens, clip_embeds, global_memory, loss, similarity
    
    def forward(self, clips, clips_large, seqs, compress_mask, qs, qs_mask, time_labels, ans_token, todo_token, insert_position, ans_position, silent_position):
        return self.forward_compress(input_ids=seqs, attention_mask=compress_mask, qs_ids=qs, qs_mask=qs_mask, images=clips, 
                                    images_large=clips_large, select_layer=self.select_layer, time_labels=time_labels, 
                                    ans_token=ans_token, todo_token=todo_token, 
                                    insert_position=insert_position, ans_position=ans_position, silent_position=silent_position)


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.compressor.dtype

    @property
    def device(self):
        return self.compressor.device

    @property
    def config(self):
        return self.compressor.config

    @property
    def hidden_size(self):
        return self.config.hidden_size


def build_compressor(compressor_cfg):
    compressor = getattr(compressor_cfg, 'mm_compressor', getattr(compressor_cfg, 'compressor', None))
    is_absolute_path_exists = os.path.exists(compressor)
    if is_absolute_path_exists:
        if 'stream' in compressor:
            return StreamQwenCompressor(compressor)
        elif 'qwen' in compressor:
            return QwenCompressor(compressor)
        else:
            return PhiCompressor(compressor)
    
    raise ValueError(f'Unknown compressor: {compressor}')


def build_compress_projector(config):
    projector_type = getattr(config, 'compress_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.compress_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.compress_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    raise ValueError(f'Unknown projector type: {projector_type}')


def build_clip_projector(config):
    projector_type = getattr(config, 'clip_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(4096, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(4096, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    raise ValueError(f'Unknown projector type: {projector_type}')
