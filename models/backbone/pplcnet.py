import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PPLCNet_x0_25", "PPLCNet_x0_35", "PPLCNet_x0_5", "PPLCNet_x0_75", "PPLCNet_x1_0",
    "PPLCNet_x1_5", "PPLCNet_x2_0", "PPLCNet_x2_5"
]

NET_CONFIG = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}

# NET_CONFIG = {
#     "blocks2":
#     #k, in_c, out_c, s, use_se
#     [[3, 16, 32, 1, False]],
#     "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
#     "blocks4": [[3, 64, 128, 1, False], [3, 128, 128, 1, False]],
#     "blocks5": [[3, 128, 256, 1, False], [5, 256, 256, 1, False],
#                 [5, 256, 256, 1, False], [5, 256, 256, 1, False],
#                 [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
#     "blocks6": [[5, 256, 512, 1, True], [5, 512, 512, 1, True]]
# }

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Hardswish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Hardsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=True) / 6.

class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            num_filters,
        )
        self.hardswish = Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = torch.mul(identity, x)
        return x


class PPLCNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 scale=1.0,
                 dropout_prob=0.0,
                 class_expand=1280,
                 class_num=361,):
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand

        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=1)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(
            in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
            out_channels=class_num, #self.class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.hardswish = Hardswish()
        self.dropout = nn.Dropout(dropout_prob)
        # self.last_conv = nn.Conv2d(self.class_expand, num_classes, kernel_size=1)
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # self.fc = nn.Linear(self.class_expand, class_num)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.blocks2(x)
        # print(x.shape)
        x = self.blocks3(x)
        # print(x.shape)
        x = self.blocks4(x)
        # print(x.shape)
        x = self.blocks5(x)
        # print(x.shape)
        x = self.blocks6(x)   
        # print(x.shape)    
        # x = self.avg_pool(x)
        # print(x.shape)
        x = self.last_conv(x)
        # print(x.shape)
        # x = self.hardswish(x)
        # x = self.dropout(x)
        # x = self.flatten(x)
        # print(x.shape)
        # x = self.fc(x)
        x = torch.special.expit(x)
        # print(x.shape)
        # x = F.log_softmax(x, dim=1)
        return x
        # return tuple(f)


def PPLCNet_x0_25(pretrained='', **kwargs):
    """
    PPLCNet_x0_25
    """
    model = PPLCNet(scale=0.25, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def PPLCNet_x0_35(pretrained='', **kwargs):
    """
    PPLCNet_x0_35
    """
    model = PPLCNet(scale=0.35, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def PPLCNet_x0_5(pretrained='', **kwargs):
    """
    PPLCNet_x0_5
    """
    model = PPLCNet(scale=0.5, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def PPLCNet_x0_75(pretrained='', **kwargs):
    """
    PPLCNet_x0_75
    """
    model = PPLCNet(scale=0.75, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def PPLCNet_x1_0(pretrained='', **kwargs):
    """
    PPLCNet_x1_0
    """
    model = PPLCNet(scale=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def PPLCNet_x1_5(pretrained='', **kwargs):
    """
    PPLCNet_x1_5
    """
    model = PPLCNet(scale=1.5, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def PPLCNet_x2_0(pretrained='', **kwargs):
    """
    PPLCNet_x2_0
    """
    model = PPLCNet(scale=2.0, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


def PPLCNet_x2_5(pretrained='', **kwargs):
    """
    PPLCNet_x2_5
    """
    model = PPLCNet(scale=2.5, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained), strict=False)
    return model


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    in_channels = 3
    cnn_scale = 1.0
    class_num = 3
    if cnn_scale == 1.0:
        model = PPLCNet_x1_0(in_channels=in_channels, class_num=class_num)
        print("pplc 10")
    elif cnn_scale == 0.5:
        model = PPLCNet_x0_5(in_channels=in_channels, class_num=class_num)
        print("pplc 05")
    elif cnn_scale == 0.75:
        model = PPLCNet_x0_75(in_channels=in_channels, class_num=class_num)
        print("pplc 075")
    elif cnn_scale == 0.25:
        model = PPLCNet_x0_25(in_channels=in_channels, class_num=class_num)
        print("pplc 025")
    elif cnn_scale == 0.35:
        model = PPLCNet_x0_35(in_channels=in_channels, class_num=class_num)
        print("pplc 035")
    device = 'cpu'
    device = 'cuda:0'
    model = model.to(device)
    model.eval()

    
    # print(model)
    input = torch.randn(1,3,224,224)
    input = torch.randn(1,3,304,304)
    # input = torch.randn(1,in_channels,19,19)
    if device == 'cuda:0':
        model = model.cuda()
        input = input.cuda()
    
    # print(len(y))
    # print(y.size())
    # summary(model, input)
    n = 100
    y = model(input)
    # exit()
    a = time.time()
    for i in tqdm(range(n)):
        y = model(input)
    b = time.time()
    print(f'cost time: {1000 * (b-a) / n :.5f} ms / it.')
    print(y.shape)
    # for x in y:
    #     print(x.shape)
    # print(model.out_channels)
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
