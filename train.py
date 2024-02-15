import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model_list import *
from modelMaker import ConvNetMaker

# Check if GPU is available, and if not, use the CPU


def data_loader(num_classes=10):
    # Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
    transforms_cifar = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if num_classes == 10:
        # Loading the CIFAR-10 dataset:
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transforms_cifar
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transforms_cifar
        )

    elif num_classes == 100:
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transforms_cifar
        )
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transforms_cifar
        )

    return train_dataset, test_dataset


def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    epochs,
    learning_rate,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
    device,
):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = (
                -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
            )

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = (
                soft_target_loss_weight * soft_targets_loss
                + ce_loss_weight * label_loss
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Dataloaders
    train_dataset, test_dataset = data_loader(100)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    torch.manual_seed(42)
    layer = plane_cifar100_book.get("10")

    teacher_model = ConvNetMaker(layer).to(device)
    train(teacher_model, train_loader, epochs=160, learning_rate=0.1, device=device)
    test_accuracy_deep = test(teacher_model, test_loader, device)

    """
    layer = plane_cifar100_book.get("2")
    torch.manual_seed(42)
    student_model = ConvNetMaker(layer).to(device)

    train(student_model, train_loader, epochs=160, learning_rate=0.1, device=device)
    test_accuracy_light_ce = test(student_model, test_loader, device)

    # Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
    kd_student_model = ConvNetMaker(layer).to(device)
    train_knowledge_distillation(
        teacher=teacher_model,
        student=kd_student_model,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        soft_target_loss_weight=0.25,
        ce_loss_weight=0.75,
        device=device,
    )
    test_accuracy_light_ce_and_kd = test(kd_student_model, test_loader, device)
    
    # Compare the student test accuracy with and without the teacher, after distillation
    """
    layer = plane_cifar100_book.get("4")
    TA_model = ConvNetMaker(layer).to(device)

    layer = plane_cifar100_book.get("2")
    takd_student_model = ConvNetMaker(layer).to(device)

    train_knowledge_distillation(
        teacher=teacher_model,
        student=TA_model,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        soft_target_loss_weight=0.25,
        ce_loss_weight=0.75,
        device=device,
    )
    train_knowledge_distillation(
        teacher=TA_model,
        student=takd_student_model,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        soft_target_loss_weight=0.25,
        ce_loss_weight=0.75,
        device=device,
    )

    test_accuracy_light_ce_and_takd = test(takd_student_model, test_loader, device)
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    # print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
    # print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
    print(f"Student accuracy with CE + TAKD: {test_accuracy_light_ce_and_takd:.2f}%")
