import torch
from collections import OrderedDict
class CNN(torch.nn.Sequential):
    def __init__(self,):
        d = []
        d.append(
            ('conv1',torch.nn.Conv2d(2,4,3)))
        d.append(
            ('relu1',torch.nn.ReLU(inplace =True)))
        
        d.append(
            ('conv2',torch.nn.Conv2d(4,1,3)))
        super().__init__(OrderedDict(d))
def main():
    torch.manual_seed(0)
    cnn = CNN()
    
    optimizer = torch.optim.Adam(cnn.parameters())
    for ib in range(10): 
        x = torch.zeros((1,2,5,5))
        y = cnn(x)
        loss = y.sum()
        
        sttdict = dict(
            input = x,
            output = y.detach(),
            loss = loss.detach(),
            **cnn.state_dict(),
        )
        torch.save(sttdict,f'sttdict_{ib}.pth')
        loss.backward()
        optimizer.step()
        print(f'loss = {loss.item()}')

if __name__ == '__main__':
    main()