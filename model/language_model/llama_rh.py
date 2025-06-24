from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from MoChat.model.arch_rh import MetaModel, MoChatMetaForCausalLM, DIoULoss


class CustomCausalLMOutputWithPastTime(CausalLMOutputWithPast):
    def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None, preds=None):
        super().__init__(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states, attentions=attentions)
        self.preds = preds

class MoChatConfig(LlamaConfig):
    model_type = "MoChat"


class MoChatLlamaModel(MetaModel, LlamaModel):
    config_class = MoChatConfig

    def __init__(self, config: LlamaConfig):
        super(MoChatLlamaModel, self).__init__(config)

class OffsetPredictor(nn.Module):
    def __init__(self, hidden_size,dropout_rate=0.5):
        super(OffsetPredictor, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.compress = nn.Linear(hidden_size, hidden_size)  # 假设 compress 是一个线性层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size*2, 2),
            nn.Dropout(dropout_rate)
        )

    def forward(self, llm_hidden_states, skeleton_features):
        # 准备查询（query）、键（key）和值（value）
        query = skeleton_features.permute(1, 0, 2)  # [maxFrame, B, hidden_size]
        key = llm_hidden_states.permute(1, 0, 2)  # [maxLength, B, hidden_size]
        value = llm_hidden_states.permute(1, 0, 2)  # [maxLength, B, hidden_size]

        # 计算交叉注意力
        attn_output, attn_weights = self.attention(query, key, value)  # attn_output: [maxFrame, B, hidden_size]
        attn_output = attn_output.permute(1, 0, 2)  # [B, maxFrame, hidden_size]
        
        # 注意力权重的形状为 [B, maxLength, maxFrame]
        attn_weights = attn_weights.permute(0,2,1)  # [B, maxFrame, 1]
        attn_weights = attn_weights[:,0,:].unsqueeze(2).permute(0,2,1)  # [B,1, maxFrame]

        weighted_skeleton_features = torch.bmm(attn_weights,skeleton_features)  # [B, 1, hidden_size]

        offsets = F.relu(self.mlp(weighted_skeleton_features.squeeze(1))) # [B, 2]
        #print(skeleton_features.size(1))
        return offsets
    
class MoChatLlamaForCausalLM(LlamaForCausalLM, MoChatMetaForCausalLM):
    config_class = MoChatConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MoChatLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pad_token_id = 0
        self.model_max_length = 2048
        self.diou_loss = DIoULoss(loss_weight=5)
        self.offset_predictor = OffsetPredictor(config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        skeletons: Optional[torch.FloatTensor] = None,
        gt_sections: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CustomCausalLMOutputWithPastTime]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, skeleton_features = self.prepare_inputs_labels_for_skeleton(input_ids, attention_mask, past_key_values, labels, skeletons)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, #这里的mask是为了避免attention到padding的token，而causal mask在model里面实现了
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        preds = self.offset_predictor(hidden_states,skeleton_features)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            llama_loss=loss_fct(shift_logits, shift_labels)
            reg_loss=self.diou_loss(preds,gt_sections)
            #print(llama_loss.item(),reg_loss.item())
            loss = llama_loss+reg_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CustomCausalLMOutputWithPastTime(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            preds=preds
        )

    # 推理时用于迭代生成
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "skeletons": kwargs.get("skeletons", None),
            }
        )
        return model_inputs

AutoConfig.register("MoChat", MoChatConfig)
AutoModelForCausalLM.register(MoChatConfig, MoChatLlamaForCausalLM)
