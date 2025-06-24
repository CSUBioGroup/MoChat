
import torch
from transformers import StoppingCriteria
from MoChat.constants import SKELETON_TOKEN_INDEX

#将<skeleton>所在位置替换为SKELETON_TOKEN_INDEX，其他部分按照tokenizer处理，最后在前面加上BOS
def tokenizer_skeleton_token(prompt, tokenizer, skeleton_token_index=SKELETON_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<skeleton>')]

    test_input_ids = []
    offset = 0
    for i, chunk in enumerate(prompt_chunks):
        if i == 0 and len(chunk) > 0 and chunk[0] == tokenizer.bos_token_id:
            offset = 1
            test_input_ids.append(chunk[0])  # 处理 BOS token
            test_input_ids.extend(chunk[1:])  # 加入 BOS token 后面的 token
        else:
            test_input_ids.extend(chunk[offset:])  # 加入当前 chunk 的所有 token
        
        # 在每个 chunk 后面添加 skeleton_token_index，除了最后一个 chunk
        if i < len(prompt_chunks) - 1:
            test_input_ids.append(skeleton_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(test_input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return test_input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        # # Adjusted to handle any number of beams
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids] #将关键词的token id转移到output_ids的设备上，否则还在cpu上

        for i in range(output_ids.shape[0]):  # Loop through each beam
            for keyword_id in self.keyword_ids:
                keyword_length = keyword_id.shape[0]  # Get the length of the keyword's token IDs
                if output_ids.shape[1] >= keyword_length:  # Ensure there are enough tokens for comparison
                    # Check if the keyword is in the last generated tokens
                    if torch.equal(output_ids[i, -keyword_length:], keyword_id):
                        return True
                    outputs= self.tokenizer.batch_decode(output_ids[i, -keyword_length:], skip_special_tokens=True)[0]
                    for keyword in self.keywords:
                        if keyword in outputs:
                            return True
        return False
