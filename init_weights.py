import torch.nn as nn
def init_model_w(model:nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)