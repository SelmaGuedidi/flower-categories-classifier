import argparse
from data import load_data, set_model,train


ap = argparse.ArgumentParser(description='Train')
ap.add_argument('data_dir', default="flowers/")
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=6)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=1012)
ap.add_argument('--gpu', default=False, action='store_true')
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.3)

p = ap.parse_args()

data_dir = p.data_dir
learning_rate = p.learning_rate
architecture = p.arch
dropout = p.dropout
hidden_units = p.hidden_units
hardware = "gpu" if p.gpu else "cpu"
epochs = p.epochs
print_every = 5

print("Loading datasets : ")
trainloaders, validationloaders  = load_data(data_dir)

print("Setting up model architecture : ")
model, criterion, optimizer = set_model(architecture, dropout, hidden_units, learning_rate, hardware,epochs)

print("Training model : ")
train(trainloaders, validationloaders, model, criterion, optimizer, epochs, print_every, hardware)

