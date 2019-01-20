import torch
import torchvision
import cub200
import numpy as np
from tqdm import tqdm
#torch.cuda.set_device(1)

_num_classes = 200
path = '/home/yuqi_huo/data/CUB_200_2011'
class BCNN(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.classifier = torchvision.models.vgg16(pretrained=True).classifier
        self.classifier = torch.nn.Sequential(*list(self.classifier.children())[:-1])

    def forward(self, X):
        N = X.size()[0]
        X = self.features(X)
        X = X.view(N, -1)
        X = self.classifier(X)
        
        return X
    
def main():
    _net = torch.nn.DataParallel(BCNN()).cuda()
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
        ])
    train_data = cub200.CUB200(root=path, train=True, download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.CenterCrop(size=224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
        ])
    test_data = cub200.CUB200(root=path, train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    _net.train(False)

    X_numpy = None
    y_numpy = None
    for X, y in tqdm(train_loader):
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda(async=True))

        score = _net(X).cpu().detach().numpy()
        y = y.cpu().numpy()
        X_numpy = np.append(X_numpy, score)
        y_numpy = np.append(y_numpy, y)
    
    np.savez('VGG-D_wo_ft_train_pytorch.npz', X_numpy, y_numpy)
    
    X_numpy = None
    y_numpy = None
    for X, y in tqdm(test_loader):
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda(async=True))

        score = _net(X).cpu().detach().numpy()
        y = y.cpu().numpy()
        X_numpy = np.append(X_numpy, score)
        y_numpy = np.append(y_numpy, y)
    
    np.savez('VGG-D_wo_ft_test_pytorch.npz', X_numpy, y_numpy)
    
if __name__ == '__main__':
    main()
