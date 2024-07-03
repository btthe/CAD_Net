from layers_cls import *

config = {}
config['chanel'] = 1
config['crop_size'] = [48, 48, 48]
config['num_hard'] = 2
config['bound_size'] = 3
config['reso'] = 1
config['sizelim'] = 10.  # mm  筛选大于6mm的
config['sizelim2'] = 20
config['sizelim3'] = 35
config['aug_scale'] = False
config['pad_value'] = 0
config['augtype'] = {'flip': True, 'swap': True, 'scale': True, 'rotate': True}
config['blacklist'] = ['417','077','188','876','057','087','130','468']
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 36, kernel_size=3, padding=1),
            nn.BatchNorm3d(36),
            nn.ReLU(inplace=True),
            nn.Conv3d(36, 36, kernel_size=3, padding=1),
            nn.BatchNorm3d(36),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [3, 3, 4]
        self.featureNum_forw = [36, 64, 128, 256]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        self.drop = nn.Dropout3d(p=0.2, inplace=False)
        self.drop1 = nn.Dropout3d(p=0.5, inplace=False)
        self.out_conv = nn.Sequential(nn.Conv3d(256, 128, kernel_size=2),
                                            nn.BatchNorm3d(128),
                                            nn.ReLU(),
                                            nn.Conv3d(128, 128, kernel_size=1),
                                            nn.ReLU(),
                                            nn.AvgPool3d((2, 2, 2), stride=(2, 2, 2)))
        self.out_fc = nn.Sequential(nn.Linear(128, 1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, dropout=False):
        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        if dropout:
            out_pool = self.drop(out_pool)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        if dropout:
            out1_pool = self.drop(out1_pool)
        out2 = self.forw2(out1_pool)  # 64
        out2_pool, indices2 = self.maxpool3(out2)
        if dropout:
            out2_pool = self.drop(out2_pool)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        if dropout:
            out3_pool = self.drop(out3_pool)

        out = self.out_conv(out3_pool)
        out = self.drop1(out)
        out = out.view(out.size(0), -1)
        out = self.out_fc(out)
        out = self.sigmoid(out)

        return out


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            # print(0,m.weight.data)
            nn.init.xavier_uniform_(m.weight, gain=1)
            # print(1,m.weight.data)
def get_model():
    net = Net()
    net.apply(_initialize_weights)
    return config, net