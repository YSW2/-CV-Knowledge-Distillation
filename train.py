import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from model_list import *
from modelMaker import ConvNetMaker
from algorithm import *
import os
import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nni

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


def print_tSNE(model, dataloader):
    model.eval()
    # 최종 출력층 데이터 추출
    final_outputs = []
    labels = []

    with torch.no_grad():
        for images, label in dataloader:
            images = images.to(device)
            outputs = model(images)
            final_outputs.extend(outputs.detach().cpu().numpy())
            labels.extend(label.cpu().numpy())

    # t-SNE를 사용하여 2차원으로 매핑
    tsne = TSNE(n_components=2, random_state=0)
    final_outputs_2d = tsne.fit_transform(np.array(final_outputs))
    labels = np.array(labels)

    # 시각화
    plt.figure(figsize=(12, 10))
    colors = cm.rainbow(np.linspace(0, 1, 100))  # 100개의 클래스에 대한 색상 생성
    for i, color in enumerate(colors):
        plt.scatter(
            final_outputs_2d[labels == i, 0],
            final_outputs_2d[labels == i, 1],
            color=color,
            label=i if i % 10 == 0 else "",
        )
    plt.legend(
        markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small"
    )
    plt.title("t-SNE Visualization of CIFAR-100 Model Outputs")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("output.png")  # 그래프를 output.png 파일로 저장


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    config = nni.get_next_parameter()
    # Dataloaders
    train_dataset, test_dataset = data_loader(100)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    torch.manual_seed(23)

    new_teacher_model = ConvNetMaker(plane_cifar100_book.get("10")).to(device)
    new_teacher_path = "./new_teacher_model.pth"
    if os.path.exists(new_teacher_path):
        new_teacher_model.load_state_dict(torch.load(new_teacher_path))
        print("모델을 불러왔습니다.")
    else:
        train(
            new_teacher_model,
            train_loader,
            epochs=160,
            learning_rate=0.1,
            device=device,
        )
        torch.save(new_teacher_model.state_dict(), new_teacher_path)

    # test_accuracy_deep = test(new_teacher_model, test_loader, device)

    torch.manual_seed(42)

    teacher_model = ConvNetMaker(plane_cifar100_book.get("10")).to(device)
    teacher_path = "./teacher_model.pth"
    if os.path.exists(teacher_path):
        teacher_model.load_state_dict(torch.load(teacher_path))
        print("모델을 불러왔습니다.")
    else:
        train(teacher_model, train_loader, epochs=160, learning_rate=0.1, device=device)
        torch.save(teacher_model.state_dict(), teacher_path)

    test_accuracy_deep = test(teacher_model, test_loader, device)

    TA_model1 = ConvNetMaker(plane_cifar100_book.get("8")).to(device)
    TA_model2 = ConvNetMaker(plane_cifar100_book.get("6")).to(device)
    TA_model3 = ConvNetMaker(plane_cifar100_book.get("4")).to(device)
    takd_student_model = ConvNetMaker(plane_cifar100_book.get("2")).to(device)
    TAKD_student_path = "./TAKD_student_path"

    if os.path.exists(TAKD_student_path):
        takd_student_model.load_state_dict(torch.load(TAKD_student_path))
        print("모델을 불러왔습니다.")
    else:
        train_knowledge_distillation(
            teacher=teacher_model,
            student=TA_model1,
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            soft_target_loss_weight=0.25,
            ce_loss_weight=0.75,
            device=device,
        )
        train_knowledge_distillation(
            teacher=TA_model1,
            student=TA_model2,
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            soft_target_loss_weight=0.25,
            ce_loss_weight=0.75,
            device=device,
        )
        train_knowledge_distillation(
            teacher=TA_model2,
            student=TA_model3,
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            soft_target_loss_weight=0.25,
            ce_loss_weight=0.75,
            device=device,
        )
        train_knowledge_distillation(
            teacher=TA_model3,
            student=takd_student_model,
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            soft_target_loss_weight=0.25,
            ce_loss_weight=0.75,
            device=device,
        )
        torch.save(takd_student_model.state_dict(), TAKD_student_path)

    test_accuracy_light_ce_and_takd = test(takd_student_model, test_loader, device)

    TA_model1 = ConvNetMaker(plane_cifar100_book.get("8")).to(device)
    TA_model2 = ConvNetMaker(plane_cifar100_book.get("6")).to(device)
    TA_model3 = ConvNetMaker(plane_cifar100_book.get("4")).to(device)
    dgkd_student_model = ConvNetMaker(plane_cifar100_book.get("2")).to(device)

    DGKD_student_path = "./DGKD_student_model.pth"
    if os.path.exists(DGKD_student_path):
        dgkd_student_model.load_state_dict(torch.load(DGKD_student_path))
        print("모델을 불러왔습니다.")
    else:
        dgkd(
            student=TA_model1,
            teacher=teacher_model,
            ta_list=[],
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            lambda_=0.5,
            device=device,
        )
        print(test(TA_model1, test_loader, device))
        dgkd(
            student=TA_model2,
            teacher=teacher_model,
            ta_list=[TA_model1],
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            lambda_=0.5,
            device=device,
        )
        print(test(TA_model2, test_loader, device))
        dgkd(
            student=TA_model3,
            teacher=teacher_model,
            ta_list=[TA_model1, TA_model2],
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            lambda_=0.5,
            device=device,
        )
        print(test(TA_model3, test_loader, device))
        dgkd(
            student=dgkd_student_model,
            teacher=teacher_model,
            ta_list=[TA_model1, TA_model2, TA_model3],
            train_loader=train_loader,
            epochs=160,
            learning_rate=0.1,
            T=2,
            lambda_=0.5,
            device=device,
        )
        torch.save(dgkd_student_model.state_dict(), DGKD_student_path)
    test_accuracy_light_ce_and_DGKD = test(dgkd_student_model, test_loader, device)

    TA_model1 = ConvNetMaker(plane_cifar100_book.get("8")).to(device)
    TA_model2 = ConvNetMaker(plane_cifar100_book.get("6")).to(device)
    TA_model3 = ConvNetMaker(plane_cifar100_book.get("4")).to(device)
    residual_teaching_student_model = ConvNetMaker(plane_cifar100_book.get("2")).to(
        device
    )
    start_time = time.time()
    train_knowledge_distillation(
        teacher=teacher_model,
        TA=TA_model1,
        student=TA_model2,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        lambda_=config.get["lambda_"],
        device=device,
    )
    train_knowledge_distillation(
        teacher=TA_model1,
        TA=TA_model2,
        student=TA_model3,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        lambda_=config.get["lambda_"],
        device=device,
    )
    train_knowledge_distillation(
        teacher=TA_model2,
        TA=TA_model3,
        student=residual_teaching_student_model,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        lambda_=config.get["lambda_"],
        device=device,
    )
    residual_teaching_test_accuracy = test(
        residual_teaching_student_model, test_loader, device
    )
    end_time = time.time()

    CTKD_time = end_time - start_time

    # print_tSNE(teacher_model, test_loader)

    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    # print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
    # print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
    print(f"Student accuracy with CE + TAKD: {test_accuracy_light_ce_and_takd:.2f}%")
    print(f"Student accuracy with CE + DGKD: {test_accuracy_light_ce_and_DGKD:.2f}%")
    print(f"co_teaching_test_accuracy: {residual_teaching_test_accuracy:.2f}%")

    print(CTKD_time)
