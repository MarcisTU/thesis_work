import torch.nn.functional as F
import torch.utils.data
from torch.utils import data
import math
import torchvision
import torchvision.transforms as transforms
from torchvision.models.segmentation.fcn import FCNHead


from loader import Dataset


class Model(torch.nn.Module):
    def __init__(self, CLASS_NUM=10, in_image_channels=4, pretrained=True):
        super().__init__()

        # in_image_channels = CLASS_NUM

        self.pretrained_model = torchvision.models.densenet169(pretrained=pretrained)
        conv1 = list(self.pretrained_model.features.children())[0]
        weight_conv1_pretrained = conv1.weight.data

        conv1 = torch.nn.Conv2d(in_image_channels,
                                conv1.out_channels, kernel_size=7, stride=1,
                                padding=3, bias=False)

        # print(self.pretrained_model)
        if in_image_channels > 3:
            conv1.weight.data[:, 0:3, :, :] = weight_conv1_pretrained[:, 0:3, :, :] # pirmie 3
            # papildus kanāls
            conv1.weight.data[:, 3:in_image_channels, :, :] = weight_conv1_pretrained[:, 0:in_image_channels-3, :, :]
        else:
            conv1.weight.data[:, 0:in_image_channels, :, :] = weight_conv1_pretrained[:, 0:in_image_channels, :, :]

        is_first = True
        features = torch.nn.Sequential()
        for name, module in self.pretrained_model.features.named_children():
            if is_first:
                module = conv1
            is_first = False
            features.add_module(name, module)

        self.pretrained_model.features = features
        self.pretrained_model.classifier = torch.nn.Linear(in_features=1664,
                        out_features=CLASS_NUM)
        # print(self.pretrained_model)

    def forward(self, x):
        y_prim = self.pretrained_model.forward(x)

        # x_3channels = x.expand(x.size(0), 3, x.size(2), x.size(3))
        # z = self.backbone(x_3channels)
        #
        # z_prim = self.classifier.forward(z['out'])
        # z_prim = F.interpolate(z_prim, size=x.shape[-2:], mode='bilinear', align_corners=False)
        #
        y_prim = torch.softmax(y_prim, dim=1) # softmax nevajag, ja izmanto iebūvēto crossentropy torch.nn.crossentropy
        return y_prim


# model = Model(CLASS_NUM=4, in_image_channels=1, pretrained=False)
# model.load_state_dict(torch.load('./classificator_models/model-best_4classes.pt', map_location=torch.device('cpu')))
# model.eval()
#
# transform = torchvision.transforms.Compose([
#     transforms.Resize(64),
#     transforms.RandomCrop((64, 64)),
#     transforms.RandomHorizontalFlip()
#     ])
# # source
# loader_params = {'batch_size': 32, 'shuffle': True, 'num_workers': 4}
# target_dataset_train = Dataset('./data/fer_48_t.hdf5', mode='train', img_size=48, transform=transform)
# dataloader_train_target = data.DataLoader(target_dataset_train, **loader_params)
#
# scores = []
# # example score
# y_real = torch.ones((32,))
# for x_t, label_t in dataloader_train_target:
#     # Happiness
#     y_prim = model.forward(x_t)
#     # y = -torch.log(y_prim)
#     y_idx = torch.argmax(y_prim, dim=1)
#     acc = torch.sum(y_real == y_idx) / 32
#     print('a')
