import math
import torch
import torch.nn as nn


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_lstm(layer):
    """
    Initialises the hidden layers in the LSTM - H0 and C0.

    Input
        layer: torch.Tensor - The LSTM layer
    """
    n_i1, n_i2 = layer.weight_ih_l0.size()
    n_i = n_i1 * n_i2

    std = math.sqrt(2. / n_i)
    scale = std * math.sqrt(3.)
    layer.weight_ih_l0.data.uniform_(-scale, scale)

    if layer.bias_ih_l0 is not None:
        layer.bias_ih_l0.data.fill_(0.)

    n_h1, n_h2 = layer.weight_hh_l0.size()
    n_h = n_h1 * n_h2

    std = math.sqrt(2. / n_h)
    scale = std * math.sqrt(3.)
    layer.weight_hh_l0.data.uniform_(-scale, scale)

    if layer.bias_hh_l0 is not None:
        layer.bias_hh_l0.data.fill_(0.)


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock1d(nn.Module):
    """
    Creates an instance of a 1D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, dil=1):
        super(ConvBlock1d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad,
                               dilation=dil)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm == 'bn':
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))

        return x


class ConvBlock2d(nn.Module):
    """
    Creates an instance of a 2D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, att=None):
        super(ConvBlock2d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm2d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.att = att
        if not self.att:
            self.act = nn.ReLU()
        else:
            self.norm = None
            if self.att == 'softmax':
                self.act = nn.Softmax(dim=-1)
            elif self.att == 'global':
                self.act = None
            else:
                self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.conv1)
        else:
            init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.att:
            x = self.conv1(x)
            if self.act():
                x = self.act(x)
        else:
            if self.norm == 'bn':
                x = self.act(self.bn1(self.conv1(x)))
            else:
                x = self.act(self.conv1(x))

        return x


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)        

        return x

def lstm_with_attention(net_params):
    if 'LSTM_1' in net_params:
        arguments = net_params['LSTM_1']
    else:
        arguments = net_params['GRU_1']
    if 'ATTENTION_1' in net_params and 'ATTENTION_Global' not in net_params:
        if arguments[-1]:
            return 'forward'
        else:
            return 'whole'
    if 'ATTENTION_1' in net_params and 'ATTENTION_Global' in net_params:
        if arguments[-1]:
            return 'forward'
        else:
            return 'whole'
    if 'ATTENTION_1' not in net_params and 'ATTENTION_Global' in net_params:
        if arguments[-1]:
            return 'forward_only'
        else:
            return 'forward_only'


def reshape_x(x):
    """
    Reshapes the input 'x' if there is a dimension of length 1

    Input:
        x: torch.Tensor - The input

    Output:
        x: torch.Tensor - Reshaped
    """
    dims = x.dim()
    if x.shape[1] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
        x = torch.reshape(x, (x.shape[0], 1))
    elif dims == 4:
        first, second, third, fourth = x.shape
        if second == 1:
            x = torch.reshape(x, (first, third, fourth))
        elif third == 1:
            x = torch.reshape(x, (first, second, fourth))
        else:
            x = torch.reshape(x, (first, second, third))
    elif dims == 3:
        first, second, third = x.shape
        if second == 1:
            x = torch.reshape(x, (first, third))
        elif third == 1:
            x = torch.reshape(x, (first, second))

    return x


class CustomMel1(nn.Module):
    def __init__(self):
        super(CustomMel1, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel2(nn.Module):
    def __init__(self):
        super(CustomMel2, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel3(nn.Module):
    def __init__(self):
        super(CustomMel3, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel4(nn.Module):
    def __init__(self):
        super(CustomMel4, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel5(nn.Module):
    def __init__(self):
        super(CustomMel5, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=5,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel6(nn.Module):
    def __init__(self):
        super(CustomMel6, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel7(nn.Module):
    def __init__(self):
        super(CustomMel7, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel8(nn.Module):
    def __init__(self):
        super(CustomMel8, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel9(nn.Module):
    def __init__(self):
        super(CustomMel9, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel10(nn.Module):
    def __init__(self):
        super(CustomMel10, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=5,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel11(nn.Module):
    def __init__(self):
        super(CustomMel11, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel12(nn.Module):
    def __init__(self):
        super(CustomMel12, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel13(nn.Module):
    def __init__(self):
        super(CustomMel13, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel14(nn.Module):
    def __init__(self):
        super(CustomMel14, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel15(nn.Module):
    def __init__(self):
        super(CustomMel15, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=5,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw1(nn.Module):
    def __init__(self):
        super(CustomRaw1, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=1024,
                                 stride=512,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)

        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw2(nn.Module):
    def __init__(self):
        super(CustomRaw2, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=512,
                                 stride=256,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)
        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw3(nn.Module):
    def __init__(self):
        super(CustomRaw3, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=1024,
                                 stride=512,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)

        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw4(nn.Module):
    def __init__(self):
        super(CustomRaw4, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=512,
                                 stride=256,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)
        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x
