import torch as th

from torch import Tensor
from torch.nn import RNNCellBase
from typing import Optional

from src.utils.enum_types import NormType


class CustomGRUCell(RNNCellBase):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 norm_type: NormType,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)
        self.norm_i = NormType.get_norm_layer_1d(norm_type, hidden_size * 3)
        self.norm_h = NormType.get_norm_layer_1d(norm_type, hidden_size * 3)
        self.norm_n = NormType.get_norm_layer_1d(norm_type, hidden_size)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = th.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        return self.gru_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

    def gru_cell(self, inputs, hidden, w_ih, w_hh, b_ih, b_hh):
        gi = self.norm_i(th.mm(inputs, w_ih.t()))
        gh = self.norm_h(th.mm(hidden, w_hh.t()))
        if self.bias:
            gi = gi + b_ih
            gh = gh + b_hh
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = th.sigmoid(i_r + h_r)
        inputgate = th.sigmoid(i_i + h_i)
        newgate = th.tanh(self.norm_n(i_n + resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy