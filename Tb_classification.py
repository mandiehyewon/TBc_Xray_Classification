from glob import glob
import numpy as np
import os
import pdb
import argparse
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from PIL import Image

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffles


#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Hyperparameters
num_epochs = 80
learning_rate = 0.001

#sort out Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, help="model you want to train", default="resnet18")
parser.add_argument("-pretrained", type=str, help="pretrained or not", default=False)
parser.add_argument("-dataset", type=str, help="dataset directory", default='ChinaSet_AllFiles')

args = parser.parse_args()
args_dict = args.__dict__

# Dataset
class TBDataset(Dataset):
	def __init__(self, datadir=args_dict["dataset"]):
		self.dir = os.path.join('/st2/hyewon/dataset/TBc/', datadir, 'CXR_png')
		self.img_path = os.listdir(self.dir)
		self.transform = self.get_transform()

	def __len__(self):
		return len(self.img_path)

	def __getitem__(self, i):
		name = self.img_path[i]
		path = os.path.join(self.dir, name)
		image = self.transform(Image.open(path).convert('RGB'))
		#pdb.set_trace()
		label = int(name[-5])

		return image, label

	def get_transform(self):
		transform = []
		transform.append(transforms.Resize((224, 224)))
		transform.append(transforms.Pad(padding=5, fill=0, padding_mode='constant'))
		transform.append(transforms.RandomCrop(224))
		transform.append(transforms.RandomHorizontalFlip())
		transform.append(transforms.ToTensor())
		transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225]))
		transform = transforms.Compose(transform) #*transforms
		
		return transform

# Data loader
dataset = TBDataset()
train_size = int(0.7*len(dataset))
test_size = len(dataset)-train_size

trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size]) 

train_loader = DataLoader(dataset=trainset,
						  batch_size=64, 
						  shuffle=True)
test_loader = DataLoader(dataset=testset,
						 batch_size=64, 
						 shuffle=False)

print('Set dataloader..........')

# Define model
class TBModel(nn.Module):
	def __init__(self):
		super(TBModel, self).__init__()

#		self.input = args_dict["model"]
		#if self.model == 'resnet18':
		#self.model = models.resnet18(pretrained=args_dict["pretrained"])
		#self.model = models.alexnet(pretrained=args_dict["pretrained"])
		#self.model = models.googlenet(pretrained=args_dict["pretrained"])
		self.model = models.densenet121(pretrained=args_dict["pretrained"])

		self.tb_fc = nn.Linear(1024, 2)
		self.maxpool = nn.MaxPool2d(7)

	def forward(self, x):
		for n, layer in self.model.named_children():
			# print(n, x.size())
			if n == 'fc':
				x = x.view(x.size(0), -1) # x size: [B, C, W, H] -> [B, -1]
				x  = self.tb_fc(x)
			elif n == "classifier":
			    x = self.maxpool(x)
			    x = x.view(x.size(0),-1)
			    x  = self.tb_fc(x)
			else:
				x = layer(x)
		return x


model = TBModel()
model.to(device)
print('Set model..........')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):	
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


# Train the model
print('Start training..........')
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)
		
		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if (i+1) % 1 == 0:
			print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
				   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

	# Decay learning rate
	if (epoch+1) % 20 == 0:
		curr_lr /= 3
		update_lr(optimizer, curr_lr)


# Test the model
model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		if total > 1000:
			break

	print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
