import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)

from transformers import AutoConfig, GPT2Model, GPT2PreTrainedModel

from .pipe import Pipe
from .partition import WithDevice, _retrieve_device
from .model import GPT2ModelCustom, GPT2LMHeadModelCustom

class ExtractFirstItem(nn.Module):
    def __init__(self):
        super(ExtractFirstItem, self).__init__()
    
    def forward(self, x):
        return x[0]

class GPT2ModelParallel(GPT2ModelCustom):
    def __init__(self, config):
        super().__init__(config)

    # ASSIGNMENT 4.2
    def _prepare_pipeline_parallel(self, split_size=1):
        '''
        Prepare the model for pipeline parallelism.

        Hint:
        1. Enable self.pipeline_parallel
        2. Construct an nn.Sequential module for the transformer layers (self.h).
        3. Use Pipe to parallelize the transformer layers.

        Please note that when implementing _prepare_pipeline_parallel, you would want to define the nn.Sequential module to extract useful values from the returned tuple. GPT2Block returns a tuple, not a tensor. 
        You should construct nn.Sequential using GPT2Block modules. Notice that each block returns multiple values but you will only need the hidden states.
        '''

        # BEGIN SOLUTION
        # Enable pipeline parallel
        self.pipeline_parallel = True
        
        # GPT2Block returns a tuple (hidden_states, ...), but we only need hidden_states
        # Create a wrapper to extract only the hidden states from each block's output
        class BlockWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
                
            def forward(self, hidden_states):
                outputs = self.block(hidden_states)
                return outputs[0]  # Return only the hidden states
        
        # Create a sequential module with wrapped blocks
        blocks_seq = nn.Sequential(*[BlockWrapper(block) for block in self.h])
        
        # Create a Pipe module with the sequential blocks
        pipe = Pipe(blocks_seq, split_size=split_size)
        
        # Set self.h_pp to the pipe
        self.h_pp = pipe
        
        return pipe
        # END SOLUTION

class GPT2LMHeadModelParallel(GPT2LMHeadModelCustom):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config, GPT2ModelParallel(config))

    def _prepare_pipeline_parallel(self, split_size=1):
        self.parallelize()
        self.transformer._prepare_pipeline_parallel(split_size)

    def _finalize_pipeline_parallel(self):
        self.deparallelize()
        self.transformer.pipeline_parallel = False

if __name__ == '__main__':
    config = AutoConfig.from_pretrained('gpt2')
    model = GPT2LMHeadModelParallel(config=config).to('cuda:0')
    model._prepare_pipeline_parallel()
