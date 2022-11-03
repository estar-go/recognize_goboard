import torch
import torch.nn.functional as F


class OnehotCrossEntropy(torch.nn.Module):

    def __init__(self):
        super(OnehotCrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = F.softmax(x, dim=1)
        y = F.one_hot(y, 361)
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1), dim=0)
        return loss
