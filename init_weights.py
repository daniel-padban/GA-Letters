import torch.nn as nn
def init_model_w(model:nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)