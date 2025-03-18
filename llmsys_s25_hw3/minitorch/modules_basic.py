"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN YOUR SOLUTION
        weights_np = np.random.normal(0, 1, (self.num_embeddings, self.embedding_dim))
        self.weights = Parameter(tensor_from_numpy(weights_np, backend=backend, requires_grad=True))
        ### END YOUR SOLUTION
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        
        # use one_hot function to convert x to one_hot
        one_hot_x = one_hot(x, self.num_embeddings)

        # flatten one_hot_x
        one_hot_x_flat = one_hot_x.view(bs * seq_len, self.num_embeddings)

        # multiply one_hot_x with self.weights
        output = one_hot_x_flat @ (self.weights.value)

        # reshape output to (bs, seq_len, embedding_dim)
        output = output.view(bs, seq_len, self.embedding_dim)
            
        return output

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN YOUR SOLUTION
        if self.training and self.p_dropout != 0:
            mask_np = np.random.binomial(1, 1-self.p_dropout, x.shape)
            mask = tensor_from_numpy(mask_np, backend=x.backend)
            return x * mask / (1 - self.p_dropout)
        else:
            return x
        ### END YOUR SOLUTION


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        x: shape (batch_size, in_size)
        W: shape (in_size, out_size)
        b: shape (out_size,)
        y: shape (batch_size, out_size)
        """
        self.out_size = out_size
        ### BEGIN YOUR SOLUTION
        bound = 1 / np.sqrt(in_size)
        weights_np = np.random.uniform(-bound, bound, (in_size, out_size))
        self.weights = Parameter(tensor_from_numpy(weights_np, backend=backend, requires_grad=True))
        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_size,))
            self.bias = Parameter(tensor_from_numpy(bias_np, backend=backend, requires_grad=True))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        # with open('minitorch_debug.log', 'a') as f:
        #     f.write(f"Linear forward called with input shape: {x.shape}\n")
        #     f.write(f"Weights shape: {self.weights.value.shape}\n")
        ### BEGIN YOUR SOLUTION
        y=x @ self.weights.value
        if self.bias:
            y=y+self.bias.value.view(1, self.out_size)
        return y
        # return (
        #     self.weights.value.view(1, in_size, self.out_size)
        #     * x.view(batch, in_size, 1)
        # ).sum(1).view(batch, self.out_size) + self.bias.value.view(1, self.out_size) if self.bias else (
        #     self.weights.value.view(1, in_size, self.out_size)
        #     * x.view(batch, in_size, 1)
        # ).sum(1).view(batch, self.out_size)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(ones((dim,), backend=backend))
        self.bias = Parameter(zeros((dim,), backend=backend))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        y = γ * x_norm + β
        """
        batch, dim = x.shape
        ### BEGIN YOUR SOLUTION
        mean = x.sum(1) / dim
        var = ((x - mean.view(batch, 1)) ** 2).sum(1) / dim
        x_norm = (x - mean.view(batch, 1)) / ((var + self.eps) ** 0.5).view(batch, 1)
        return self.weights.value.view(1, dim) * x_norm + self.bias.value.view(1, dim)
        ### END YOUR SOLUTION

class LayerNorm(Module):
    def __init__(self, dim: int, backend: TensorBackend):
        super().__init__()

        self.gamma = Parameter(zeros((dim,), backend=backend))
        self.beta = Parameter(zeros((dim,), backend=backend))
    
    def forward(self, x: Tensor) -> Tensor:
        return x.layernorm(self.gamma.value, self.beta.value)
    
class Attn_Softmax(Module):
    def __init__(self, mask: Tensor, backend: TensorBackend):
        super().__init__()

        self.mask = mask
    
    def forward(self, x: Tensor) -> Tensor:
        return x.attn_softmax(self.mask)
    