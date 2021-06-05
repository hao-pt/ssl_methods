import torch
import torchvision.models as models

from resnet import ShakeResNet
import wrn


def get_wide_resnet(dropout_rate, num_classes, pretrained=True):
    # model = models.wide_resnet50_2(pretrained)
    model = wrn.wrn_40_2(dropout_rate, num_classes)
    return model

def get_resnet(pretrained=True):
    model = models.resnet18(pretrained)
    return model

def get_shake_resnet(in_ch=3, 
    depth=26, width=96, num_classes=10, pretrained=False):
    out_chs = [16, width, width*2, width*4] # output channels of 4 blocks
    model = ShakeResNet(in_ch, 
        out_chs, 
        depth, 
        num_classes)

    return model

MODEL_ZOO = {
    "wide_resnet50_2": get_wide_resnet,
    "resnet18": get_resnet,
    "shake_resnet26": get_shake_resnet
}

def get_model(model_name, pretrained=True, 
    ema=False, num_classes=10):
    model_getter = MODEL_ZOO[model_name]
    model = model_getter(num_classes=num_classes, pretrained=pretrained)

    # detach params
    if ema:
        for param in model.parameters():
            param.detach_()

    return model

if __name__ == "__main__":
    model_name = "shake_resnet26"
    model = get_model(model_name, pretrained=True)
    print(model)
    