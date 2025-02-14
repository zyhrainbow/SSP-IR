from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor, Attention
import os

import numpy as np
import PIL.Image

from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
)
from diffusers.utils.torch_utils import randn_tensor
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class BIM(nn.Module):
    def __init__(self,
                 dropout=0.0,
                 num_degrade=5,
                 prompt_channels=77,
                 prompt_dim = 1024,
                 hidden_size = 512,
                 num_heads = 8,
                 device = 'cuda',
                 input_dim = 1024,
                 refine = True
                ):
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.prompt_channels = prompt_channels
        self.prompt_dim = prompt_dim
        self.num_degrade = num_degrade
        self.refine = refine

        self.learnable_param = nn.Parameter(torch.rand(num_degrade, prompt_channels,prompt_dim))
        self.transform_in = nn.Sequential(nn.Linear(input_dim, prompt_dim),
                                          nn.LayerNorm(prompt_dim))
                                        #    nn.Tanh(), 
                                        #    nn.Linear(prompt_dim, prompt_dim)) 
        self.weight_predictor = nn.Sequential(nn.Linear(input_dim, num_degrade),
                                             nn.Tanh(),
                                             nn.Linear(num_degrade, num_degrade),
                                             nn.Softmax(dim=1))
       
        num_head_channels = hidden_size // num_heads

        self.atten = Attention(hidden_size,hidden_size,heads=num_heads,dim_head=num_head_channels)

        self.transform_out = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                           nn.Tanh(), 
                                           nn.Linear(hidden_size, prompt_dim))
        self.attn1 = Attention(prompt_dim, prompt_dim, heads=num_heads, dim_head=num_head_channels)
        self.fc = nn.Sequential(nn.Linear(prompt_dim, prompt_dim),
                                nn.Tanh(),
                                nn.Linear(prompt_dim, prompt_dim),
                                nn.Tanh(),
                                nn.Linear(prompt_dim, prompt_dim))
        
        if self.refine:
            self.attn2 = Attention(prompt_dim, prompt_dim, heads=num_heads, dim_head=num_head_channels)
            
     
            
        
        #      
        
    def forward(self, img_emb_o, img_emb_g):
        # img_emb [B,257,1024] img_emb_g [B,1024]
        
        B, N, C = img_emb_o.shape
        weight = self.weight_predictor(img_emb_g.squeeze(1)).unsqueeze(2).unsqueeze(2) #[B,num_degrade, 1, 1]
        img_emb_o = self.transform_in(img_emb_o)
        img_emb = img_emb_o.repeat(self.num_degrade, 1, 1) #[B*num_degrade, 257, 1024]
        pred_emb = self.learnable_param.repeat(B, 1, 1)#[B*num_degrade, 257, 1024]
        pred_emb = self.attn1(pred_emb,img_emb).view(B, self.num_degrade, self.prompt_channels, self.prompt_dim )#[B, num_degrade, 257, 1024]
        pred_emb = pred_emb * weight
        pred_emb = torch.sum(pred_emb, dim=1)
        pred_emb = self.fc(pred_emb)
        
        if self.refine:
            img_emb_o = self.attn2(img_emb_o, pred_emb)
           
            
        return pred_emb, img_emb_o  
     

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def pad_tokens(tokens, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]

    return tokens


def get_unweighted_text_embeddings(
    text_encoder,
    text_input: torch.Tensor,
    chunk_length: int=77,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = text_encoder(text_input_chunk)[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings = text_encoder(text_input)[0]
    return text_embeddings


def get_text_index(
    tokenizer,
    prompt: Union[str, List[str]],
    max_embeddings_multiples: Optional[int] = 4,
    no_boseos_middle: Optional[bool] = False,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
    """
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt_tokens = [
        token[1:-1] for token in tokenizer(prompt, max_length=max_length, truncation=True).input_ids
    ]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = getattr(tokenizer, "pad_token_id", eos)
    prompt_tokens = pad_tokens(
        prompt_tokens,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)

    return prompt_tokens   
# device = "cuda"
# model = BIM().to(device)
# img_emb = torch.rand(2,257, 1024).to(device)
# img_emb_g = torch.rand(2, 1024).to(device)
# pred_emb = model(img_emb, img_emb_g)

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
    
class TPBNet(nn.Module):
    def __init__(self, num_cross_proj_layers: int = 2, clip_v_dim: int = 1024, last_dim: int=768):
        super().__init__()

        layer_list = []
        for i in range(num_cross_proj_layers):
            layer_list +=[nn.Linear(clip_v_dim, clip_v_dim), nn.LayerNorm(clip_v_dim), nn.LeakyReLU()]
        layer_list += [nn.Linear(clip_v_dim, last_dim)]
        self.visual_projection = nn.Sequential(*layer_list)


    def forward(
        self,
        clip_vision_outputs: Optional[torch.FloatTensor] = None,
        use_global: Optional[bool] = False,
        layer_ids: Optional[List[int]] = [24],
        batch_index: Optional[int] = None,
    ):
        # convert layer_ids to list
        if isinstance(layer_ids, int):
            layer_ids = [layer_ids]
        if len(layer_ids) > 1:
            # TODO: support multiple layers
            pass
        else:
            layer_id = layer_ids[0]
            assert layer_id >= 0 and layer_id < 25, "layer_id must be in [0, 24]"
            if use_global:
                # projection_input = clip_vision_outputs.hidden_states[layer_id]
                projection_input = clip_vision_outputs.pooler_output.unsqueeze(1)
            else:
                if batch_index is not None:
                    projection_input = clip_vision_outputs.hidden_states[layer_id][batch_index, 1:, :].unsqueeze(0)
                else:
                    projection_input = clip_vision_outputs.hidden_states[layer_id][:, 1:, :]

        image_embeds = self.visual_projection(projection_input)

        return image_embeds