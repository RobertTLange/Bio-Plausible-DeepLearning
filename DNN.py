import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger, update_logger

# Fully connected neural network with one hidden layer
class DNN(nn.Module):
    def __init__(self, h_sizes=[784, 500], out_size=10):
        super(DNN, self).__init__()
        self.layers = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(h_sizes[k+1], out_size))

        self.print_architecture()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def print_architecture(self):
        for layer in self.layers:
            print(layer)


def train_dnn_model(model, num_epochs,
                    train_loader, test_loader,
                    device, optimizer, criterion,
                    model_fname ="temp_model.ckpt",
                    verbose=True, logging=True):
    logger = Logger('./logs')

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()

            if (i+1) % 100 == 0:
                # Set model to eval mode for dropout and batch norm
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = images.reshape(-1, 28*28).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                test_accuracy = float(correct)/total
                # Set model back to train mode
                model.train()

                # Save the model checkpoint
                torch.save(model.state_dict(), model_fname)
                if verbose:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Acc: {:.2f}, Test Acc: {:.2f}'
                           .format(epoch+1, num_epochs,
                                   i+1, len(train_loader),
                                   loss.item(), accuracy.item(), test_accuracy))

                if logging:
                    update_logger(logger, epoch, i, loss, accuracy, model,
                                  images, train_loader)


if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define batchsize for data-loading
    batch_size = 100

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Feedforward Neural Network Parameters
    num_epochs = 5

    # Instantiate the model with layersize and Logging directory
    dnn_model = DNN(h_sizes=[784, 500, 300, 100], out_size=10).to(device)
    logger = Logger('./logs')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)

    train_dnn_model(dnn_model, num_epochs,
                    train_loader, test_loader,
                    device, optimizer, criterion,
                    model_fname ="models/temp_model_dnn.ckpt",
                    verbose=True, logging=True)
