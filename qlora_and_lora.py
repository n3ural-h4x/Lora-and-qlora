#!pip install bitsandbytes
#first install bitsandbytes in your venv
import torch
import torch.nn.functional as F
import torch.nn as nn
import bitsandbytes as bnb
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Qlora_4_bit(nn.Module):
  def __init__(self, in_features:int, out_features:int, alpha:int, bias:int, r:int, quantize_weight:bool, require_lora:bool, use_qlora:bool):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = alpha
    self.r_in = r
    self.bias = bias
    self.scale = self.alpha / self.r_in
    self.use_qlora = use_qlora
    self.require_lora = require_lora
    self.quant_nn = bnb.nn.Linear4bit(self.in_features,
                                              self.out_features,
                                              bias= None if not self.bias else self.bias,
                                              compute_dtype=torch.bfloat16,
                                              compress_statistics=False) if self.use_qlora else nn.Linear(self.in_features, self.out_features, bias=None if not self.bias else True)
    self.W_a = nn.Parameter(torch.randn(self.in_features, self.r_in))
    self.W_b = nn.Parameter(torch.randn(self.r_in, self.out_features))

    #IMP_SANITY CHECK:
    self.module_has_been_quantized = False

    if quantize_weight:
     self.quant_nn.weight.requires_grad = False

    #INTIALISZE THE WEIGHTS:
    #torch.nn.init.
    torch.nn.init.zeros_(self.W_b)
    torch.nn.init.kaiming_uniform_(self.W_a, a=math.sqrt(5))

  def forward(self, x:torch.Tensor):

    inter_m = self.quant_nn(x)

    if self.require_lora:

     casted_W_a = self.W_a.to(x.dtype)
     casted_W_b = self.W_b.to(x.dtype)
     output = inter_m + self.scale * ((x @ casted_W_a) @ casted_W_b)
     return output
    else:
     return inter_m

def quantize_model(module, mlp_layers:list, alpha:int, r:int, require_lora:bool, use_qlora:bool):
  for name, child in list(module.named_children()):
    if isinstance(child, nn.Linear) and name in mlp_layers and not hasattr(child, 'module_has_been_quantized'):

      new_layer = Qlora_4_bit(
          child.in_features,
          child.out_features,
          alpha=alpha,
          bias=child.bias is not None,
          r=r,
          quantize_weight=True,
          require_lora=require_lora,
          use_qlora=use_qlora
          ).to(device) # set to cpu if you cpu

      with torch.no_grad():
          new_layer.quant_nn.weight.data = child.weight.data.clone()
          if child.bias is not None:
            new_layer.quant_nn.bias.data = child.bias.data.clone()
          new_layer.module_has_been_quantized = True
      setattr(module, name, new_layer)
    else:
      quantize_model(child, mlp_layers, alpha, r, require_lora, use_qlora)
