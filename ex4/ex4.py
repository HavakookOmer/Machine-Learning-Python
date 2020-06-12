import torch.nn as nn
import torch.nn.functional as F
from gcommand_loader import GCommandLoader
import torch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16280, 1000)
        self.fc2 = nn.Linear(1000, 30)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


def do_back_propagation(loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Train the model
def train_data(train_loader,optimizer, model):
    model.train()
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward-propagation pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.nll_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Example [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))


def print_func(correct, total):
    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))


# Test the model
def test_data(test_loader, model):
    print("in test_data")
    # Test the model
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print_func(correct, total)


# Write to file the results of the test
def write_to_file(test_loader, test_x, model):
    list_commands = []
    list_y_hat = []
    enter = '\n'
    comma = ', '
    zero = 0
    one = 1

    file = open("test_y", "w")

    # the path of each file
    for x in test_x:
        x = x[zero]
        x = x.split("/")
        y = x[len(x)-one]
        list_commands.append(y)

    # the prediction of each file
    for voices, labels in test_loader:
        outputs = model(voices)
        _, y_hat = torch.max(outputs.data, 1)
        list_y_hat.extend(y_hat.tolist())

    # write each path of file and it's prediction into the file
    for x, y in zip(list_commands, list_y_hat):
        file.write(x + comma + str(y) + enter)
    file.close()


if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    num_classes = 30
    len_image = 101
    len_image_2 = 161
    image_size = len_image * len_image_2
    size_of_batch = 100

    # get the data-set
    dataset = GCommandLoader('ML4_dataset/data/train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=size_of_batch, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    # get the validation-set
    validation_set = GCommandLoader('ML4_dataset/data/valid')

    valid_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=size_of_batch, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    # get the test-set
    test_set = GCommandLoader('ML4_dataset/data/test')

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=size_of_batch, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = LeNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # for the back-propagation

    # train the model
    train_data(train_loader, optimizer, model)

    # test the model
    test_data(valid_loader, model)
    #
    # # write to file the results of the test
    # write_to_file(test_loader, test_set.spects, model)