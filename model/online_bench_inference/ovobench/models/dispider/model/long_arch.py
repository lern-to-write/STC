#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from .language_model.builder import build_compressor, build_compress_projector, build_clip_projector

from models.dispider.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LongMetaModel:

    def __init__(self, config):
        super(LongMetaModel, self).__init__(config)
        
        if hasattr(config, "mm_compressor"):
            self.compressor = build_compressor(config)
            self.compress_projector = build_compress_projector(config)
            self.clip_projector = build_clip_projector(config)

    def get_compressor(self):
        compressor = getattr(self, 'compressor', None)
        return compressor

    def initialize_compress_modules(self, model_args):
        compressor = model_args.compressor
        pretrain_compress_mlp_adapter = model_args.pretrain_compress_mlp_adapter
        pretrain_clip_mlp_adapter = model_args.pretrain_clip_mlp_adapter
        pretrain_dual_mlp_adapter = model_args.pretrain_dual_mlp_adapter

        self.config.mm_compressor = compressor

        if self.get_compressor() is None:
            self.compressor = build_compressor(model_args)

        self.config.use_compress_proj = True
        self.config.compress_projector_type = getattr(model_args, 'compress_projector_type', 'linear')
        self.config.compress_hidden_size = self.compressor.hidden_size
        self.config.clip_projector_type = getattr(model_args, 'clip_projector_type', 'linear')
        self.config.clip_hidden_size = 4096

        if getattr(self, 'compress_projector', None) is None:
            self.compress_projector = build_compress_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.compress_projector.parameters():
                p.requires_grad = True

        self.clip_projector = build_clip_projector(self.config)
        # In case it is frozen by LoRA
        for p in self.clip_projector.parameters():
            p.requires_grad = True

        if pretrain_compress_mlp_adapter is not None:
            compress_projector_weights = torch.load(pretrain_compress_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.compress_projector = build_compress_projector(self.config)
            self.compress_projector.load_state_dict(get_w(compress_projector_weights, 'compress_projector'), strict=True)
            self.compressor.compressor.load_state_dict(get_w(compress_projector_weights, 'compressor.compressor'))

        if pretrain_clip_mlp_adapter is not None:
            clip_projector_weights = torch.load(pretrain_clip_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.clip_projector = build_clip_projector(self.config)
            self.clip_projector.load_state_dict(get_w(clip_projector_weights, 'clip_projector'), strict=True)

        if pretrain_dual_mlp_adapter is not None:
            dual_projector_weights = torch.load(pretrain_dual_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.compress_projector = build_compress_projector(self.config)
            self.compress_projector.load_state_dict(get_w(dual_projector_weights, 'compress_projector'), strict=True)
            self.clip_projector = build_clip_projector(self.config)
            self.clip_projector.load_state_dict(get_w(dual_projector_weights, 'clip_projector'), strict=True)


class LongMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_compressor(self):
        return self.get_model().get_compressor()
    
    def encode_sequences(self, clips, clips_large, seqs, compress_mask, qs, qs_mask, time_labels, ans_token, todo_token, insert_position, ans_position, silent_position, random_list):
        clip_features, clip_embeds, global_memory, loss, similarity = self.get_model().get_compressor()(clips, clips_large, seqs, compress_mask, qs, qs_mask, time_labels, ans_token, todo_token, insert_position, ans_position, silent_position)
        if type(clip_features) is list:
            if random_list is None:
                random_list = [i for i in range(len(clip_features))]
            concat_clip_features = [clip_features[i] for i in random_list]
            concat_clip_features = self.get_model().compress_projector(torch.cat(concat_clip_features, dim=0)) # qk k c
            concat_clip_embeds = [clip_embeds[i] for i in random_list]
            concat_clip_embeds = self.get_model().clip_projector(torch.cat(concat_clip_embeds, dim=0)) # qk k c
            concat_global_memory = [global_memory[i] for i in random_list]
            concat_global_memory = self.get_model().compress_projector(torch.cat(concat_global_memory, dim=0)) # qm c
            concat_clip_features = torch.cat([concat_clip_embeds, concat_clip_features], dim=-2) # qk n c
            clip_index = 0
            memory_index = 0
            new_clip_features = []
            for i in random_list:
                clip_feature = concat_clip_features[clip_index: clip_index+clip_features[i].shape[0]]
                clip_feature = clip_feature.view(-1, clip_feature.shape[-1])
                partial_memory = concat_global_memory[memory_index: memory_index+global_memory[i].shape[0]]
                new_clip_features.append(torch.cat([clip_feature, partial_memory], dim=0))
                clip_index = clip_index + clip_features[i].shape[0]
                memory_index = memory_index + global_memory[i].shape[0]
            clip_features = new_clip_features
        else:
            clip_features = self.get_model().compress_projector(clip_features) # q 4 k c
            clip_embeds = self.get_model().clip_projector(clip_embeds) # q 4 k c
            global_memory = self.get_model().compress_projector(global_memory)
            clip_features = torch.cat([clip_embeds, clip_features], dim=-2) # stage 3
            clip_features = clip_features.view(clip_features.shape[0], -1, clip_features.shape[-1])
            clip_features = torch.cat([clip_features, global_memory], dim=-2)
        return clip_features, loss, similarity

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, clips, clips_large, seqs, compress_mask, qs, qs_mask, time_labels, ans_token, todo_token, insert_position, ans_position, silent_position
    ):
        compressor = self.get_compressor()
        if compressor is None or clips is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, 0, 0 # llama-3
            if isinstance(past_key_values, tuple) and compressor is not None and clips is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            elif past_key_values is not None and past_key_values.seqlen_offset>0:
                target_shape = past_key_values.seqlen_offset + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, 0, 0
        
        random_list = None
        # if input_ids.shape[0] > 3:
        #     random_list = torch.randperm(input_ids.shape[0])[:3]
        #     input_ids = input_ids[random_list]
        #     labels = labels[random_list]
        #     attention_mask = attention_mask[random_list]
        image_features, loss, similarity = self.encode_sequences(clips, clips_large, seqs, compress_mask, qs, qs_mask, time_labels, ans_token, todo_token, insert_position, ans_position, silent_position, random_list)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                try:
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                except:
                    cur_input_embeds_1 = self.get_model().tok_embeddings(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            try:
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            except:
                cur_input_embeds = self.get_model().tok_embeddings(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # print('finish preparing labels multimodal')

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, loss, similarity

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
