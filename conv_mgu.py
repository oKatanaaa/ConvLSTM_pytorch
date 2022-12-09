import torch.nn.functional as F
import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        std = torch.std(x, dim=[1, 2, 3], keepdim=True)
        mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        x = (x - mean) / (std + 1e-4)
        return x


class ConvMGUCell(nn.Module):
    # https://arxiv.org/pdf/1603.09420.pdf
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, n_groups=4):
        """
        Initialize ConvMGUCell cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvMGUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        self.candidate_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.hidden_state = None
        nn.init.orthogonal_(self.candidate_conv.weight, gain=nn.init.calculate_gain('tanh'))
        
        self.gate_conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        nn.init.orthogonal_(self.gate_conv.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(self.gate_conv.bias, val=-2.0)
        self.init_hidden()
        self.gn = nn.GroupNorm(num_groups=n_groups, num_channels=self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        # Broadcast [1, с, 1, 1] -> [b, c, h, w]
        b, _, h, w = input_tensor.shape
        h_cur = cur_state.expand([b, -1, h, w])
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        f = torch.sigmoid(self.gate_conv(combined))
        
        combined2 = torch.cat([input_tensor, f * h_cur], dim=1)
        h_c_pretanh = self.candidate_conv(combined2)
        h_c_pretanh = self.gn(h_c_pretanh)
        h_c = F.tanh(h_c_pretanh)
        
        h_next = (1. - f) * h_cur + f * h_c
        
        return h_next

    def init_hidden(self):
        if self.hidden_state is None:
            # [bs, c, h, w]
            self.hidden_state = nn.Parameter(
                torch.randn(1, self.hidden_dim, 1, 1, device=self.gate_conv.weight.device))
        return self.hidden_state


class ConvMGU(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of MGU layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvMGU(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False, n_groups=4):
        super(ConvMGU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvMGUCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias, n_groups=n_groups))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, reverse=False):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list: list
            [[h, c] * n_layers], where h and c are the last step tensors ([bs, c, h, w]).
            If return_all_layers=True returns values ([[h, c]]) for last layer only.
        layer_output:
            [h * n_layers], where h is stacked along time dimension ([t, bs, c, h, w])
            If return_all_layers=True returns values ([h]) for last layer only.
        """
        if self.batch_first:
            # Prioritizing time dimension gives a slight boost in performance
            # (b, t, c, h, w) -> (t, b, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is None:
            hidden_state = self._init_hidden()

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor
        
        start, end, step = 0, seq_len, 1
        if reverse:
            start, end, step = seq_len - 1, -1, -1

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(start, end, step):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[t],
                                                 cur_state=h)
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output

            layer_output_list.append(
                layer_output.permute(1, 0, 2, 3, 4).contiguous() if self.batch_first else layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden())
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
