import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import cv2 as cv
import numpy as np


class VGG(nn.Module):

    def __init__(self):
        super().__init__()

        self.block_size = [2, 2, 3, 3, 3]

        # 3 input image channel, 64 output channels, 3x3 square convolution
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1)

        # 64 feature maps again to 64 feature maps
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding = 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding = 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding = 1)

        self.FC6 = nn.Linear(512 * 7 * 7, 4096) # 7 * 7 from image dimension
        self.FC7 = nn.Linear(4096, 4096)
        self.FC8 = nn.Linear(4096, 2622) # 2622 classes

    def forward(self, x):

        # input x.dim = (224, 224, 3)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, (2, 2)) # max pooling, window size of (2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, (2, 2))        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, (2, 2))

        # flatten the feature maps: (1, N), where -1 does the job to calculate N
        x = x.view(x.size(0), -1)

        # FCs
        x = F.relu(self.FC6(x))
        x = F.dropout(x, 0.5, self.training)
        x7 = F.relu(self.FC7(x))
    
        x8 = F.dropout(x7, 0.5, self.training)
        return(x7, self.FC8(x8))

    # num_flat_features() could do the flattening, but not necessary.
    '''
    def num_flat_features(self, x): # so maybe it is flatten() in keras
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    '''


    def load_weights(self, path="/home/hyobin/Documents/SoSe20/SHK/vgg_face_torch/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "FC%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

if __name__ == "__main__":
    model = VGG().double()
    model.load_weights()

    im = cv.imread("/home/hyobin/Documents/SoSe20/SHK/vgg_face_torch/ak.png")
    # change the dimension format from opencv to torch
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224).double()
    
    model.eval()
    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
    preds = F.softmax(model(im), dim=1)
    values, indices = preds.max(-1)
    print("values: ", values, "indices: ", indices)