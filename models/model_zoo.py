# from models.sseg.deeplabv3plus import DeepLabV3Plus
# from models.sseg.fcn import FCN
#from models.sseg.light_head import LightHead
from models.sseg.pspnet import PSPNet
from models.sseg.deeplabv3plus import DeepLabV3Plus
from models.sseg.spnet import SPNet
# from thop import profile
def get_model(model, backbone, pretrained, nclass, lightweight):
    model_instance = None
    if model == 'spnet':
        model_instance = SPNet(backbone, pretrained, nclass, lightweight)
    else:
        print("\nError: MODEL \'%s\' is not implemented!\n" % model)
        exit(1)

    #参数量计算params
    params_num = sum(p.numel() for p in model_instance.parameters())
    print("\nParams: %.1fM" % (params_num / 1e6))

    return model_instance



