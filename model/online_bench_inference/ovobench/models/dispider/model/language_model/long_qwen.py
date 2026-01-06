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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
# from .qwen_model.modeling_qwen import *
# from .qwen_model.configuration_qwen import Qwen2Config
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..long_arch import LongMetaModel, LongMetaForCausalLM
from transformers.generation.utils import GenerateOutput

class LongConfig(Qwen2Config):
    model_type = "long"


class LongQwen2Model(LongMetaModel, Qwen2Model):
    config_class = LongConfig

    def __init__(self, config: Qwen2Config):
        super(LongQwen2Model, self).__init__(config)


class LongQwen2ForCausalLM(Qwen2ForCausalLM, LongMetaForCausalLM):
    config_class = LongConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LongQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_large: Optional[torch.FloatTensor] = None,
        seqs: Optional[torch.LongTensor] = None,
        compress_mask: Optional[torch.Tensor] = None,
        qs: Optional[torch.LongTensor] = None,
        qs_mask: Optional[torch.Tensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        projector: Optional[torch.LongTensor] = None,
        q_id: Optional[str] = None,
        insert_position: Optional[int] = None,
        ans_position: Optional[list] = None,
        silent_position: Optional[list] = None,
        ans_token: Optional[torch.LongTensor] = None,
        todo_token: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                time_loss,
                similarity
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_large,
                seqs,
                compress_mask,
                qs,
                qs_mask,
                time_labels,
                ans_token,
                todo_token,
                insert_position,
                ans_position,
                silent_position
            )
            
        # if inputs_embeds.shape[0] > 3:
        #     random_list = torch.randperm(inputs_embeds.shape[0])[:3]
        #     inputs_embeds = inputs_embeds[random_list]
        #     labels = labels[random_list]
        #     attention_mask = attention_mask[random_list]

        res =  super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if self.training:
            res.loss = res.loss + time_loss

        return res

    @torch.no_grad() # qwen-2
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_large: Optional[torch.FloatTensor] = None,
        seqs: Optional[torch.LongTensor] = None,
        compress_mask: Optional[torch.Tensor] = None,
        qs: Optional[torch.LongTensor] = None,
        qs_mask: Optional[torch.Tensor] = None,
        time_labels: Optional[torch.FloatTensor] = None,
        projector: Optional[torch.LongTensor] = None,
        q_id: Optional[str] = None,
        insert_position: Optional[int] = None,
        ans_position: Optional[list] = None,
        silent_position: Optional[list] = None,
        ans_token: Optional[torch.LongTensor] = None,
        todo_token: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                time_loss,
                similarity
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_large,
                seqs,
                compress_mask,
                qs,
                qs_mask,
                time_labels,
                ans_token,
                todo_token,
                insert_position,
                ans_position,
                silent_position
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        
        # if time_loss.item() < -1:
        #     return time_loss

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_large = kwargs.pop("images_large", None)
        seqs = kwargs.pop("seqs", None)
        compress_mask = kwargs.pop("compress_mask", None)
        qs = kwargs.pop("qs", None)
        qs_mask = kwargs.pop("qs_mask", None)
        q_id = kwargs.pop("q_id", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if images_large is not None:
            _inputs['images_large'] = images_large
        if seqs is not None:
            _inputs['seqs'] = seqs
        if compress_mask is not None:
            _inputs['compress_mask'] = compress_mask
        if qs is not None:
            _inputs['qs'] = qs
        if qs_mask is not None:
            _inputs['qs_mask'] = qs_mask
        if q_id is not None:
            _inputs['q_id'] = q_id
        _inputs['time_labels'] = None
        return _inputs

AutoConfig.register("long", LongConfig)
AutoModelForCausalLM.register(LongConfig, LongQwen2ForCausalLM)
