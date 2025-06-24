from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from MoChat.constants import *

class MetaModel:
    def __init__(self, config):
        super(MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"): 
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_skeleton_encoder(self, model_args, fsdp=None):
        
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.mm_vision_tower = model_args.vision_tower

        skeleton_tower = build_vision_tower(model_args)

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = 48*23 
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [skeleton_tower]
        else:
            self.vision_tower = skeleton_tower

        if not hasattr(self, 'mm_projector') or not self.mm_projector.weight.size(0):
            self.mm_projector = nn.Linear(self.config.mm_hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

class MoChatMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def encode_skeletons(self, skeletons):
        skeleton_features = self.get_model().get_vision_tower()(skeletons)
        return skeleton_features

    def prepare_inputs_labels_for_skeleton(self, input_ids, attention_mask, past_key_values, labels, skeletons):
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or skeletons is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and skeletons is not None and input_ids.shape[1] == 1: #推理生成时更新mask长度
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        
        skeleton_features = self.encode_skeletons(skeletons) #[16, 300, 1056] [16, 512]
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_skeleton_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            skeleton_token_indices = torch.where(cur_input_ids == SKELETON_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            if skeleton_token_indices.numel() > 0:
                cur_skeleton_features = []
                skeleton_token_start = skeleton_token_indices[0]
                skeleton_token_end = skeleton_token_indices[-1]

                for _ in skeleton_token_indices:
                    cur_skeleton_features.append(skeleton_features[cur_skeleton_idx])
                    cur_skeleton_idx += 1

                cur_skeleton_features = torch.stack(cur_skeleton_features, dim=0) #将[300,1056]变为[1,300,1056]
                cur_skeleton_features = self.get_model().mm_projector(cur_skeleton_features) #[1,300,5120]
                if cur_skeleton_features.dim() == 3:
                    t, l, n = cur_skeleton_features.size()
                    cur_skeleton_features = cur_skeleton_features.contiguous().view(t * l, n) #[300,5120]

                #添加前半段和骨架特征
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:skeleton_token_start]))

                cur_new_input_embeds.append(cur_skeleton_features)

                if labels is not None:
                    cur_new_labels.append(cur_labels[:skeleton_token_start])
                    cur_new_labels.append(torch.full((cur_skeleton_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) #忽略输入的骨架特征
                    cur_labels = cur_labels[skeleton_token_end+1:]
                cur_input_ids = cur_input_ids[skeleton_token_end+1:]
            
            #添加后半段文字
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]#确保cur_new_input_embeds中的每个元素都在同一个设备上
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0) #拼接前半段、骨架特征和后半段
            new_input_embeds.append(cur_new_input_embeds)#将单个样本的cur_new_input_embeds添加到new_input_embeds中
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        pad_token_embedding = self.get_model().embed_tokens(torch.tensor([self.pad_token_id], device=cur_skeleton_features.device)).detach()

        # 将不等长的 new_input_embeds 填充为等长，并构造 attention_mask
        max_length = max([emb.shape[0] for emb in new_input_embeds])
        padded_new_input_embeds = []
        attention_masks = []

        for emb in new_input_embeds:
            emb = emb.to(cur_skeleton_features.device)  # 确保 emb 在目标设备上
            padding_length = max_length - emb.shape[0]
            padding_tensor = pad_token_embedding.repeat(padding_length, 1)
            padded_emb = torch.cat([emb, padding_tensor], dim=0)
            padded_new_input_embeds.append(padded_emb)
            
            # 构造 attention_mask
            attention_mask_prepare = torch.cat([torch.ones(emb.shape[0], dtype=attention_mask.dtype, device=attention_mask.device), torch.zeros(padding_length, dtype=attention_mask.dtype, device=attention_mask.device)])
            attention_masks.append(attention_mask_prepare)

        # 转换为张量
        padded_new_input_embeds = torch.stack(padded_new_input_embeds)
        attention_masks = torch.stack(attention_masks)

        # 截断到模型最大长度
        padded_new_input_embeds = padded_new_input_embeds[:, :self.model_max_length, :]
        attention_masks = attention_masks[:, :self.model_max_length]

        if labels is not None:
            padded_new_labels = torch.nn.utils.rnn.pad_sequence(
                new_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
            new_labels = padded_new_labels[:, :self.model_max_length]

        if attention_mask is not None:          
            attention_mask = attention_masks
            assert attention_mask.shape == padded_new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, padded_new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):

        if model_args.use_frameid_token:
            tokenizer.add_tokens([f'<frameid_{i}>' for i in range(model_args.max_frame)]) #不考虑time cls token
            self.resize_token_embeddings(len(tokenizer))
        self.pad_token_id = tokenizer.pad_token_id
        self.model_max_length = tokenizer.model_max_length
