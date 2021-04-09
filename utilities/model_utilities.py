import torch
import torch.nn as nn


def create_tensor_data(x, cuda):
    """
    Converts the data from numpy arrays to torch tensors

    Inputs
        x: The input data
        cuda: Bool - Set to true if using the GPU

    Output
        x: Data converted to a tensor
    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def calculate_loss(prediction, target, cw=None, gender=True):
    """
    With respect to the final layer of the model, calculate the loss of the
    model.

    Inputs
        prediction: The output of the model
        target: The relative label for the output of the model
        cw: torch.Tensor - The class weights for the dataset
        gender: bool set True if splitting data according to gender

    Output
        loss: The BCELoss or NLL_Loss
    """
    if gender:
        if target.shape[0] != cw.shape[0]:
            fem_nd_w, fem_d_w, male_nd_w, male_d_w = cw
            zero_ind = (target == 0).nonzero().reshape(-1)
            one_ind = (target == 1).nonzero().reshape(-1)
            two_ind = (target == 2).nonzero().reshape(-1)
            three_ind = (target == 3).nonzero().reshape(-1)
            class_weights = torch.ones(target.shape[0])
            class_weights.scatter_(0, zero_ind, fem_nd_w[0])
            class_weights.scatter_(0, one_ind, fem_d_w[0])
            class_weights.scatter_(0, two_ind, male_nd_w[0])
            class_weights.scatter_(0, three_ind, male_d_w[0])
            cw = class_weights.reshape(-1, 1)
        target = target % 2
        if type(cw) is not torch.Tensor:
            cw = torch.Tensor(cw)
    else:
        if type(cw) is not torch.Tensor:
            cw = torch.Tensor(cw)
        if target.shape[0] != cw.shape[0]:
            zero_ind = (target == 0).nonzero().reshape(-1)
            one_ind = (target == 1).nonzero().reshape(-1)
            class_weights = torch.ones(target.shape[0])
            class_weights.scatter_(0, zero_ind, cw[0])
            class_weights.scatter_(0, one_ind, cw[1])
            cw = class_weights.reshape(-1, 1)

    if prediction.dim() == 1:
        prediction = prediction.view(-1, 1)

    bceloss = nn.BCELoss(weight=cw)
    loss = bceloss(prediction, target.float().view(-1, 1))

    return loss
