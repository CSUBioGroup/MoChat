import math
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=150):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Attention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0., proj_do_rate=0.):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head ** -0.5

        self.qkv = nn.Linear(dim_emb, dim_emb * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_do_rate)
        self.proj = nn.Linear(dim_emb, dim_emb)  
        self.proj_dropout = nn.Dropout(proj_do_rate)

    def forward(self, x, mask=None):

        B, N, C = x.shape  #(b f) j c or b f (j c)

        qkv = self.qkv(x)#[B,N,C*3]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask_value = float('-inf')  # 设置为-inf会自动根据单精度和半精度的不同进行调整，无需手动-1e9或者-1e4
            attn = attn.masked_fill(mask == 0, mask_value) #mask shape:[B,1,f,f], attn shape:[B,num_heads,N,N]，需要进行广播

        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn) #非常规的dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#pytorch的transpose只能交换两个维度，多维度交换需要用permute
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, do_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_joint=23, num_frame=300, dim_emb=48, 
                num_heads=8, ff_expand=1.0, qkv_bias=False, attn_do_rate=0., proj_do_rate=0., drop_path=0., positional_emb_type='learnalbe'):

        super(TransformerEncoder, self).__init__()

        self.positional_emb_type = positional_emb_type

        # for learnable positional embedding
        self.positional_emb = nn.Parameter(torch.zeros(1, num_frame+1, num_joint, dim_emb))

        # for fixed positional embedding (ablation)
        self.tm_pos_encoder = PositionalEmbedding(num_joint*dim_emb, num_frame+1)
        self.sp_pos_encoder = PositionalEmbedding(dim_emb, num_joint)#dim_emb是最后一个维度，num_joint是倒数第二维度即位置编码对应0到num_joint-1的空间位置

        self.norm1_sp = nn.LayerNorm(dim_emb)
        self.norm1_tm = nn.LayerNorm(dim_emb*num_joint)

        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)
        self.attention_tm = Attention(dim_emb*num_joint, num_heads, qkv_bias, attn_do_rate, proj_do_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim_emb*num_joint)
        self.feedforward = FeedForward(in_features=dim_emb*num_joint, hidden_features=int(dim_emb*num_joint*ff_expand), 
                                        out_features=dim_emb*num_joint, do_rate=proj_do_rate)
                            

    def forward(self, x, s_mask=None,t_mask=None, positional_emb=False):

        b, f, j, c = x.shape #这里的f和j都加了cls

        ## spatial-MHA
        x_sp = rearrange(x, 'b f j c  -> (b f) j c', )
        x_sp = x_sp + self.drop_path(self.attention_sp(self.norm1_sp(x_sp), mask=s_mask)) #drop_path随机丢弃一些残差连接
  
        ## temporal-MHA
        x_tm = rearrange(x_sp, '(b f) j c -> b f (j c)', b=b, f=f)
        x_tm = x_tm + self.drop_path(self.attention_tm(self.norm1_tm(x_tm), mask=t_mask))

        x_out = x_tm
        x_out = x_out + self.drop_path(self.feedforward(self.norm2(x_out)))
        x_out = rearrange(x_out, 'b f (j c)  -> b f j c', j=j) #分裂的时候必须提供至少一个维度的大小

        return x_out

class ST_Transformer(nn.Module):

    def __init__(self, max_frame, max_joint, input_channel, dim_joint_emb,
                depth, num_heads, qkv_bias, ff_expand, do_rate, attn_do_rate,
                drop_path_rate, add_positional_emb, positional_emb_type):

        super(ST_Transformer, self).__init__()

        self.add_positional_emb = add_positional_emb
        
        self.dropout = nn.Dropout(p=do_rate)
        self.norm = nn.LayerNorm(dim_joint_emb*max_joint)

        self.emb=nn.Linear(input_channel, dim_joint_emb)
        self.emb_global = nn.Linear(input_channel, dim_joint_emb)
        self.emb_arms = nn.Linear(input_channel, dim_joint_emb)
        self.emb_legs = nn.Linear(input_channel, dim_joint_emb)
        self.emb_spine = nn.Linear(input_channel, dim_joint_emb)

        self.tm_fuse_token = nn.Parameter(torch.rand(1,1,max_joint,dim_joint_emb))

        # for learnable positional embedding
        self.positional_emb = nn.Parameter(torch.zeros(1, max_frame+1, max_joint, dim_joint_emb))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]#递增的drop_path_rate

        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(max_joint, max_frame, dim_joint_emb, 
            num_heads, ff_expand, qkv_bias, attn_do_rate, do_rate, dpr[i], positional_emb_type) #do_rate是proj_do_rate, dpr[i]是drop_path_rate
            for i in range(depth)]
        )
        
        self.mlp = nn.Sequential(
                                nn.Linear(dim_joint_emb*max_joint, dim_joint_emb*max_joint),
                                nn.GELU(),
                                nn.LayerNorm(dim_joint_emb*max_joint),
                                nn.Linear(dim_joint_emb*max_joint, dim_joint_emb*max_joint),
                                nn.GELU(),
                                nn.LayerNorm(dim_joint_emb*max_joint),
                                )

    def encoder(self, x, s_mask=None, t_mask=None):

        b, c, f, j_cls = x.shape

        ## Joints-Grouped Embedding
        center_joint = 22
        x_global = x[:,:,:,[center_joint,]]
        x_arms = x[:,:,:,[16,18,20,17,19,21]]#左右手
        x_legs = x[:,:,:,[1,4,7,10,2,5,8,11]]#左右腿
        x_spine = x[:,:,:,[0,3,6,9,12,13,14,15]]#脊柱

        x_global = rearrange(x_global, 'b c f j -> b f j c')
        x_arms = rearrange(x_arms, 'b c f j -> b f j c')
        x_legs = rearrange(x_legs, 'b c f j -> b f j c')
        x_spine = rearrange(x_spine, 'b c f j -> b f j c')

        #joints和global分开映射，也就会分开训练
        x_arms = self.emb_arms(x_arms)
        x_legs = self.emb_legs(x_legs)
        x_spine = self.emb_spine(x_spine)
        x_global = self.emb_global(x_global)  # 作为cls token

        x = torch.cat((x_arms,x_legs,x_spine,x_global), axis=2)
        new_order = [16, 18, 20, 17, 19, 21, 
        1, 4, 7, 10, 2, 5, 8, 11,  
        0, 3, 6, 9, 12, 13, 14, 15, 
        22] 
        x = torch.cat((x,self.tm_fuse_token.repeat(b,1,1,1)),dim=1) #[B,F+1,J*P+1,C]
        
        x = self.dropout(x)

        x_sp = rearrange(x, 'b f j c  -> (b f) j c', )
        pos_emb = self.positional_emb.repeat(b, 1,1,1) #learnable positional embedding，复制batch_size份
        pos_emb = rearrange(pos_emb, 'b f j c -> (b f) j c', b=b, f=f+1)
        x_sp = x_sp + pos_emb

        x_tm = rearrange(x_sp, '(b f) j c -> b f (j c)', b=b, f=f+1)
        pos_emb = rearrange(pos_emb, '(b f) j c -> b f (j c)', b=b, f=f+1)
        x_tm = x_tm + pos_emb
        x=rearrange(x_tm, 'b f (j c) -> b f j c', j=j_cls)
        
        # 创建逆序索引，用于将 x_combined 排列成与原始 x 相同的顺序
        inverse_order = sorted(range(len(new_order)), key=lambda k: new_order[k])
        x=x[:,:,inverse_order,:]

        ## GL-Transformer blocks
        for i, block in enumerate(self.encoder_blocks):
            if self.add_positional_emb:
                positional_emb=True
            else:
                positional_emb = False
            x = block(x, s_mask, t_mask, positional_emb)

        x = rearrange(x, 'b f j k -> b f (j k)')
        x = self.norm(x)
        tm_cls=x[:,-1,:]
        x_tm=x[:,:-1,:]
        x = rearrange(x_tm, 'b f (j k) -> b f j k',j=j_cls)
        sp_cls=x[:,:,-1,:]
        
        return x


    def forward(self, x):
        B, C, T, VM = x.shape #此时的vm是（joint+1）
        # 提取帧数和关节点数维度的数据
        x_frames_joint0 = x[:, :, :, 0]  # shape: [B, C, T]
        x_joints_frame0 = x[:, :, 0, :]  # shape: [B, C, V*M]

        # 创建掩码 同时给cls留空间
        t_mask = (x_frames_joint0 != 99.9).all(dim=1, keepdim=True).to(x.device)  # shape: [B, 1, T]
        s_mask = (x_joints_frame0 != 99.9).all(dim=1, keepdim=True).to(x.device)  # shape: [B, 1, V*M]
        joint_num_cls = s_mask[0].sum().item() #假设所有样本的关节数相同
        if(joint_num_cls==VM and s_mask.all()):
            #print('无需移动')
            pass
        else:
            # 移动关节点cls到最后
            x_joints=torch.cat([x[:, :, :, :joint_num_cls-1], x[:, :, :, joint_num_cls:]], dim=-1)
            x = torch.cat([x_joints, x[:, :, :, joint_num_cls-1].unsqueeze(3)], dim=3)
            s_mask[:,0,joint_num_cls-1]=False
            s_mask[:,0,VM-1]=True  # shape: [B, 1, V*M]

        # 给t_mask增加CLS token位置
        t_cls = torch.ones(B, 1, 1, dtype=torch.bool, device=x.device) 

        # 拼接在最后
        t_mask = torch.cat([t_mask, t_cls], dim=2)  # shape: [B, 1, T+1]

        t_mask = t_mask.repeat(1, T+1, 1).unsqueeze(1)
        s_mask = s_mask.repeat(T+1, VM, 1).unsqueeze(1) #因为后续要把B和F合并，所以要复制T+1份

        x = self.encoder(x, s_mask, t_mask)

        ## MLP
        x = rearrange(x, 'b f j k -> b f (j k)',j=VM)
        x = self.mlp(x) 
        x = rearrange(x, 'b f (j k) -> b f j k',j=VM)
        x = torch.cat([x[:, :, :joint_num_cls-1,:], x[:,:,-1,:].unsqueeze(2), x[:, :, joint_num_cls-1:-1, :]], dim=2)
        x = rearrange(x, 'b f j k -> b f (j k)',j=VM)

        # 提取前 T 个 mask
        mask = t_mask[:,:, :T, :T]
        filtered_samples = []
        for i in range(B):
            # 取得第 i 个样本和对应的 mask
            sample = x[i]
            sample_mask = mask[i].squeeze(0)[0]
            valid_indices = sample_mask.nonzero(as_tuple=True)[0]
            # 筛选样本
            filtered_sample = sample[valid_indices]
            filtered_samples.append(filtered_sample)

        return filtered_samples
        
class JGSE(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.transformer = ST_Transformer(max_frame=500, max_joint=23, input_channel=3, dim_joint_emb=48,
                depth=4, num_heads=8, qkv_bias=True, ff_expand=2.0, do_rate=0.1, attn_do_rate=0.1,
                drop_path_rate=0.1, add_positional_emb=1, positional_emb_type='learnable')
        self.transformer.load_state_dict(torch.load(model_path))
        self.transformer.requires_grad_(False)
        self.transformer.eval() #dropout和batchnorm在eval和train的时候行为不一样

        #兼容CLIP
        self.image_processor=None
        self.image_eval_processor=None
    
    @property
    def device(self):
        # 假设模型的参数都在同一个设备上
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        # 假设模型的所有参数都有相同的数据类型
        return next(self.parameters()).dtype

    def forward(self, x):
        B, C, T, V, M = x.shape
        x = x.contiguous().view(B,C,T,V*M)
        return self.transformer(x.to(device=self.device, dtype=self.dtype)) #[B,F,J*K]