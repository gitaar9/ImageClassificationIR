import torch

from feature_extracting_pretrained_resnets import feature_extracting_resnet50
from pytorch_vais_dataset import VAISDataset

ds = VAISDataset(root_dir='/home/gitaar9/TNO_Thesis/ImageClassificationIR/datasets/VAIS')

# resnet = models.resnet152(pretrained=True)
resnet = feature_extracting_resnet50(pretrained=True)
print(dir(resnet))
# print(resnet.predict(ds[0]))
print(resnet(torch.tensor(ds[0][0])))
