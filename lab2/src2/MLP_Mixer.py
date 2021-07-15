import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

# 禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        # 这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        # self.skip_connection = ?
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(patch_size, patch_size),
            nn.GELU(),
            nn.Linear(patch_size, patch_size),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        ########################################################################

    def forward(self, x):
        ########################################################################
        y0 = torch.transpose(self.layernorm(x), 0, 1)
        y1 = torch.transpose(self.mlp1(y0), 0, 1)
        y2 = self.mlp2(self.layernorm(y1 + x))
        return y2
        ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        # 这里写Pre-patch Fully-connected, Global average pooling, fully connected
        self.pre_patch_fc = nn.Conv2d(hidden_dim, hidden_dim, stride=patch_size, kernel_size=patch_size * patch_size,
                                      groups=hidden_dim)
        self.mlp = Mixer_Layer(patch_size, hidden_dim)
        self.fc = nn.Linear(7, 7)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.depth = depth
        ########################################################################

    def forward(self, data):
        ########################################################################
        # 注意维度的变化
        mlp1_data = self.pre_patch_fc(data)
        mlp1_data = self.mlp(mlp1_data)
        mlp1_data = self.layernorm(mlp1_data)

        avg = torch.mean(mlp1_data, dim=1)
        mlp2_data = self.fc(avg)
        return mlp2_data
        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 计算loss并进行优化
            pred = model(data)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.
    num_correct = 0  # correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            ########################################################################
            # 需要计算测试集的loss和accuracy
            pred = model(data)
            test_loss += criterion(pred, target).item()
            num_correct += (pred.argmax(1) == target).type(torch.float).sum().item()

        test_loss /= len(test_loader)
        accuracy = num_correct / len(test_loader.dataset)
        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))


if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    ########################################################################
    # 这里需要调用optimizer，criterion(交叉熵)
    model = MLPMixer(patch_size=7, hidden_dim=128, depth=3).to(device)  # 参数自己设定，其中depth必须大于1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    ########################################################################

    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)
