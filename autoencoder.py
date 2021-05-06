import torch
import torch.nn as nn

class net_2(nn.Module):
    def __init__(self):
        super(net_2,self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.classifier = torch.nn.Sequential(torch.nn.Linear(193600, 4096),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(4096, 2048),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(p=0.5),
                                               torch.nn.Linear(2048, 2),
                                               nn.Softmax(dim=1))
    def forward(self, x):
        encoded = self.encoder(x)
        result = self.classifier(encoded)
        return result

