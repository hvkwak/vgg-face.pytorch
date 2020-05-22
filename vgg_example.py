import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import cv2 as cv
import numpy as np
import os
# import tqdm
from os import listdir
from os.path import isfile, join, isdir


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
    
def compute_descriptors(phase): # computes descriptors of Train/Test dataset
    # phase: 'Train' or 'Test'
    model = VGG()
    model.load_weights()

    mypath = "/home/hyobin/Documents/vgg-face.pytorch/images/"+phase
    people = np.sort(listdir(mypath)) # additional sort() needed.

    # number of images and memory for labels
    num_imgs = 0
    for k in range(len(people)):
        num_imgs = num_imgs + len(listdir(mypath + "/" + people[k]))
    labels = torch.zeros(num_imgs, dtype = int)
    
    # compute descriptors per person:
    for i in range(len(people)):
        print(i)
        img_path = mypath + "/" + people[i]
        n = len(listdir(img_path))
        descriptors = torch.zeros([n, 4096])
        labels[n*i:n*(i+1)] = i+1

        img_names = listdir(img_path)
        for k in range(n): # n images per person
            img_name = img_names[k]
            img = cv.imread(img_path + "/" + img_name)
            img = cv.resize(img, (224, 224))
            img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224)
            model.eval()
            img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
            descriptor = model(img)[0]
            descriptors[k, :] = descriptor
        
        torch.save(descriptors, phase + '_descriptors{}.pt'.format(i))
    torch.save(labels, phase + '_labels.pt')

def classifier(test_img):
    mypath = "/home/hyobin/Documents/vgg-face.pytorch/descriptors/descriptors/"
    n = len(listdir(mypath))

    # Startwert: Infinity
    best_distance = np.Inf
    best_label = n+1
    
    # per class compute the best Nearest Neighbor(NN)
    for i in range(n): 
        descriptor = torch.load(mypath+"Train_descriptors{}.pt".format(i))
        # compute Euclidean Distance:
        NN = torch.min(torch.sum((descriptor - test_img)**2, 1)) # row sums
        if NN < best_distance:
            best_distance = NN
            best_label = i
    return(best_distance, best_label)

def einzel_test(img_name):
    model = VGG()
    model.load_weights()
    img = cv.imread(img_name)
    img = cv.resize(img, (224, 224))
    img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224)
    model.eval()
    img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
    test_img = model(img)[0]
    __, NN_label = classifier(test_img)
    # fuer Einzeltest muss es um 1 erhoeht werden
    print(NN_label+1) 

def test():
    model = VGG()
    model.load_weights()
    testpath = "/home/hyobin/Documents/vgg-face.pytorch/images/"+"Test"
    people = np.sort(listdir(testpath)) # additional sort() needed.
    result_mat = torch.zeros((len(people), len(people)), dtype=int)

    # compute descriptors per person:
    for i in range(len(people)): # Here comes the label.
        print(i)
        img_path = testpath + "/" + people[i]
        img_names = listdir(img_path)
        n = len(img_names)
        for k in range(n): # n images per person
            img_name = img_names[k]
            img = cv.imread(img_path + "/" + img_name)
            img = cv.resize(img, (224, 224))
            img = torch.Tensor(img).permute(2, 0, 1).view(1, 3, 224, 224)
            model.eval()
            img -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
            test_img = model(img)[0]
            __, NN_label = classifier(test_img)
            result_mat[i, NN_label] = result_mat[i, NN_label] + 1
        print(result_mat)
    return(result_mat)

if __name__ == "__main__":
    print("Einzeltest Result:")
    einzel_test("/home/hyobin/Documents/vgg-face.pytorch/images/Test/20_Hyovin/00020_0303202011129.png")
    print("")
    result_mat = test()
    print(result_mat)
    print(sum(torch.diag(result_mat))/torch.sum(result_mat).double())
    # tensor(0.9961, dtype=torch.float64)
