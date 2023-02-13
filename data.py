import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

with open('cat_to_name.json', 'r') as f:
	cat_to_name = json.load(f)

def load_data(data_dir):
	data_dir = 'flowers'
	train_dir = data_dir +'/train'
	valid_dir = data_dir +'/valid'
	train_transforms = transforms.Compose([transforms.RandomRotation(60),
										   transforms.RandomResizedCrop(224),
										   transforms.RandomHorizontalFlip(),
										   transforms.ToTensor(),
										   transforms.Normalize([0.485, 0.456, 0.406],
																[0.229, 0.224, 0.225])])
	test_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

	train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
	valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)

	trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
	validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
	return trainloaders,validloaders

def set_model(architecture, dropout, hidden_units, learning_rate, hardware,epochs):
	if(architecture=="vgg16"):
		model = models.vgg16(pretrained = True)
	elif(architecture=="vgg13"):
		model = models.vgg13(pretrained = True)
	for param in model.parameters():
		param.requires_grad = False
	classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088, 2096)),
						  ('relu', nn.ReLU()),
						  ('dropout', nn.Dropout(0.3)),
						  ('fc2', nn.Linear(2096, 312)),
						  ('relu', nn.ReLU()),
						  ('fc3', nn.Linear(312, 102)),
						  ('output', nn.LogSoftmax(dim=1))
						  ]))
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	return model, criterion, optimizer

def train(trainloaders, validationloaders, model, criterion, optimizer, epochs, print_every, hardware):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	epochs = epochs
	steps = 0
	running_loss = 0
	for epoch in range(epochs):
		for inputs, labels in trainloaders:
			steps += 1
			inputs, labels = inputs.to(device), labels.to(device)
			logps = model.forward(inputs)
			loss = criterion(logps, labels)
		
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
		
			if steps % print_every == 0:
				valid_loss = 0
				accuracy = 0
				model.eval()
				with torch.no_grad():
					for inputs, labels in validloaders:
						inputs, labels = inputs.to(device), labels.to(device)
						logps = model.forward(inputs)
						batch_loss = criterion(logps, labels)
					
						valid_loss += batch_loss.item()
						ps = torch.exp(logps)
						top_p, top_class = ps.topk(1, dim=1)
						equals = top_class == labels.view(*top_class.shape)
						accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
					
				print(f"Epoch {epoch+1}/{epochs}.. "
				 	  f"Train loss: {running_loss/print_every:.3f}.. "
				  	  f"Validation loss: {valid_loss/len(validloaders):.3f}.. "
				  	  f"Validation accuracy: {accuracy*100/len(validloaders):.3f}%")
				running_loss = 0
				model.train()
	
	