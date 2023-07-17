from .flownet import FlowNet

def flownet():
    net = FlowNet(num_blocks=9, heat=1, half_layer=[3, 6], num_features=31)
    net.use_2dconv = True
    net.bandwise = False
    net.pairwise = True
    return net