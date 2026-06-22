from torch.nn import Linear, LeakyReLU, Identity
from torch.nn import Module, functional as F, Sequential
import torch

class Adapter(Module):
    def __init__(self, input_dim, output_dim=None, 
                 bottleneck=None, last_activation=True):
        super(Adapter, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        if bottleneck is not None:
            self.adapter = Sequential(
                Linear(input_dim, bottleneck),
                LeakyReLU(),
                Linear(bottleneck, output_dim),
                LeakyReLU() if last_activation else Identity()
            )
        else:
            self.adapter = Sequential(
                Linear(input_dim, output_dim),
                LeakyReLU() if last_activation else Identity()
            )


    def forward(self, x):
        return self.adapter(x)
    
if __name__ == "__main__":
    adapter = Adapter(512, 256, bottleneck=128, last_activation=False)
    x = torch.randn(1, 512)
    output = adapter(x)
    print(output.shape)