import torch
import torch.nn.functional as F
from abc import abstractmethod

from torch import nn


class FBankResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.network(x)
        out = out + x
        out = self.relu(out)
        return out

#---------------------------------------------------------------------------------------------------------------------------------------------#
class FBankNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=32, out_channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=(5 - 1)//2, stride=2),
            FBankResBlock(in_channels=64, out_channels=64, kernel_size=3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=128, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=(5 - 1) // 2, stride=2),
            FBankResBlock(in_channels=256, out_channels=256, kernel_size=3),
            nn.AvgPool2d(kernel_size=4)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(256, 250)
        )

    @abstractmethod
    def forward(self, *input_):
        raise NotImplementedError('Call one of the subclasses of this class')
#---------------------------------------------------------------------------------------------------------------------------------------------#
class FBankCrossEntropyNet(FBankNet):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_layer = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x):
        n = x.shape[0]
        out = self.network(x)
        out = out.reshape(n, -1)
        out = self.linear_layer(out)
        return out

    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val
    #---------------------------------------------------------------------------------------------------------------------------------------------#

def get_cosine_distance(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    return (1 - F.cosine_similarity(a, b)).numpy()


MODEL_PATH = './triplet_loss_trained_model.pth'
model_instance = FBankCrossEntropyNet()
model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
model_instance = model_instance.double()
model_instance.eval()


def get_embeddings(x):
    print('enetered get embeddings fct')
    x = torch.from_numpy(x)
    with torch.no_grad():
        embeddings = model_instance(x)
    print("will retuen in get embeddings fct")
    print("----------EMBEDDINGS SHAPE--------",embeddings.shape)
    return embeddings.numpy()
