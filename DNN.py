import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger

# Fully connected neural network with one hidden layer
class DNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_dnn_model(model, num_epochs,
                train_loader, test_loader,
                device, optimizer, model_fname ="temp_model.ckpt",
                verbose=True, logging=True):

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
                    # 1. Log scalar values (scalar summary)
                    info = { 'loss': loss.item(), 'accuracy': accuracy.item() }

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, epoch*len(train_loader) + i+1)

                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), epoch*len(train_loader) + i+1)
                        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch*len(train_loader) + i+1)

                    # 3. Log training images (image summary)
                    info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

                    for tag, images in info.items():
                        logger.image_summary(tag, images, epoch*len(train_loader) + i+1)


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
    dnn_model = DNN(input_size=784, hidden_size=500, num_classes=10).to(device)
    logger = Logger('./logs')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)

    train_dnn_model(dnn_model, num_epochs,
                    train_loader, test_loader,
                    device, optimizer, model_fname ="temp_model.ckpt",
                    verbose=True, logging=True)
