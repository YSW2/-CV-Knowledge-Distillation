import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from model_list import *
from modelMaker import ConvNetMaker
import os
import time
import random


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


def CTKD_teacher_to_TA(
    teacher,
    TA1,
    TA2,
    train_loader,
    epochs,
    learning_rate,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
    device,
):
    distillation_criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer1 = optim.SGD(
        TA1.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )
    optimizer2 = optim.SGD(
        TA2.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    teacher.eval()  # Teacher set to evaluation mode
    TA1.train()  # Student to train mode
    TA2.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_soft_labels = torch.softmax(teacher_logits / T, dim=1)

            # TA1과 TA2에 대해 선택된 인덱스를 사용하여 입력 처리
            TA1_outputs = TA1(inputs)
            TA2_outputs = TA2(inputs)

            optimizer1.zero_grad()
            loss1 = F.cross_entropy(TA1_outputs, labels)
            distill_loss1 = distillation_criterion(
                torch.log_softmax(TA1_outputs / T, dim=1),
                teacher_soft_labels,
            )
            total_loss1 = (
                ce_loss_weight * loss1 + soft_target_loss_weight * distill_loss1
            )
            total_loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2 = F.cross_entropy(TA2_outputs, labels)
            distill_loss2 = distillation_criterion(
                torch.log_softmax(TA2_outputs / T, dim=1),
                teacher_soft_labels,
            )
            total_loss2 = (
                ce_loss_weight * loss2 + soft_target_loss_weight * distill_loss2
            )
            total_loss2.backward()
            optimizer2.step()

            running_loss += total_loss1.item() + total_loss2.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def CTKD_TA_to_TA(
    TA_Teacher1,
    TA_Teacher2,
    TA1,
    TA2,
    train_loader,
    epochs,
    learning_rate,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
    device,
    remember_rate,
):
    distillation_criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer1 = optim.SGD(
        TA1.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )
    optimizer2 = optim.SGD(
        TA2.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    TA_Teacher1.eval()  # Teacher set to evaluation mode
    TA_Teacher2.eval()
    TA1.train()  # Student to train mode
    TA2.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                TA_Teacher1_logits = TA_Teacher1(inputs)
                TA_Teacher2_logits = TA_Teacher2(inputs)

                TA_Teacher1_soft_labels = torch.softmax(TA_Teacher1_logits / T, dim=1)
                TA_Teacher2_soft_labels = torch.softmax(TA_Teacher2_logits / T, dim=1)

            TA_Teacher1_loss = F.cross_entropy(
                TA_Teacher1_logits, labels, reduction="none"
            )
            TA_Teacher2_loss = F.cross_entropy(
                TA_Teacher2_logits, labels, reduction="none"
            )

            _, topk_indices1 = torch.topk(
                -TA_Teacher1_loss,
                k=int(len(TA_Teacher1_loss) * remember_rate),
                sorted=False,
            )
            _, topk_indices2 = torch.topk(
                -TA_Teacher2_loss,
                k=int(len(TA_Teacher2_loss) * remember_rate),
                sorted=False,
            )

            # 학생 모델에 대한 입력 선택
            selected_inputs1 = inputs[topk_indices2]
            selected_labels1 = labels[topk_indices2]
            selected_inputs2 = inputs[topk_indices1]
            selected_labels2 = labels[topk_indices1]

            TA1_outputs = TA1(selected_inputs1)
            TA2_outputs = TA2(selected_inputs2)

            optimizer1.zero_grad()
            loss1 = F.cross_entropy(TA1_outputs, selected_labels1)
            distill_loss1 = distillation_criterion(
                torch.log_softmax(TA1_outputs / T, dim=1),
                TA_Teacher1_soft_labels[topk_indices2],
            )
            total_loss1 = (
                ce_loss_weight * loss1 + soft_target_loss_weight * distill_loss1
            )
            total_loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2 = F.cross_entropy(TA2_outputs, selected_labels2)
            distill_loss2 = distillation_criterion(
                torch.log_softmax(TA2_outputs / T, dim=1),
                TA_Teacher2_soft_labels[topk_indices1],
            )
            total_loss2 = (
                ce_loss_weight * loss2 + soft_target_loss_weight * distill_loss2
            )
            total_loss2.backward()
            optimizer2.step()

            running_loss += total_loss1.item() + total_loss2.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def CTKD_TA_to_Student(
    TA1,
    TA2,
    student,
    train_loader,
    epochs,
    learning_rate,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
    device,
    remember_rate,
):
    distillation_criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    TA1.eval()  # Teacher set to evaluation mode
    TA2.eval()
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                TA1_logits = TA1(inputs)
                TA2_logits = TA2(inputs)

                TA1_soft_labels = torch.softmax(TA1_logits / T, dim=1)
                TA2_soft_labels = torch.softmax(TA2_logits / T, dim=1)

            TA1_loss = F.cross_entropy(TA1_logits, labels, reduction="none")
            TA2_loss = F.cross_entropy(TA2_logits, labels, reduction="none")

            _, topk_indices1 = torch.topk(
                -TA1_loss, k=int(len(TA1_loss) * remember_rate), sorted=False
            )
            _, topk_indices2 = torch.topk(
                -TA2_loss, k=int(len(TA2_loss) * remember_rate), sorted=False
            )

            # 학생 모델에 대한 입력 선택
            selected_inputs1 = inputs[topk_indices2]
            selected_labels1 = labels[topk_indices2]
            selected_inputs2 = inputs[topk_indices1]
            selected_labels2 = labels[topk_indices1]

            student1_outputs = student(selected_inputs1)
            student2_outputs = student(selected_inputs2)

            optimizer.zero_grad()
            loss1 = F.cross_entropy(student1_outputs, selected_labels1)
            distill_loss1 = distillation_criterion(
                torch.log_softmax(student1_outputs / T, dim=1),
                TA2_soft_labels[topk_indices2],
            )
            total_loss1 = (
                ce_loss_weight * loss1 + soft_target_loss_weight * distill_loss1
            )
            total_loss1.backward()

            loss2 = F.cross_entropy(student2_outputs, selected_labels2)
            distill_loss2 = distillation_criterion(
                torch.log_softmax(student2_outputs / T, dim=1),
                TA1_soft_labels[topk_indices1],
            )
            total_loss2 = (
                ce_loss_weight * loss2 + soft_target_loss_weight * distill_loss2
            )
            total_loss2.backward()
            optimizer.step()

            running_loss += total_loss1.item() + total_loss2.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def dgkd(
    student,
    teacher,
    ta_list,
    train_loader,
    epochs,
    learning_rate,
    T,
    lambda_,
    device,
):
    distillation_criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    teacher.eval()
    for ta in ta_list:
        ta.eval()
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            student_logits = student(inputs)
            loss_SL = F.cross_entropy(student_logits, labels)

            with torch.no_grad():
                teahcer_logits = teacher(inputs)
                ta_logits = []
                for ta in ta_list:
                    ta_logits.append(ta(inputs))

                loss_KD_list = [
                    distillation_criterion(
                        F.log_softmax(student_logits / T, dim=1),
                        F.softmax(teahcer_logits / T, dim=1),
                    )
                ]

                # Teacher Assistants Knowledge Distillation Loss
                for i in range(len(ta_list)):
                    loss_KD_list.append(
                        distillation_criterion(
                            F.log_softmax(student_logits / T, dim=1),
                            F.softmax(ta_logits[i] / T, dim=1),
                        )
                    )

            for _ in range(len(loss_KD_list) // 2):
                loss_KD_list.remove(random.choice(loss_KD_list))

            loss = (1 - lambda_) * loss_SL + lambda_ * T * T * sum(loss_KD_list)

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

    DGKD_student_path = "./DGKD_student_mode.pth"
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

    """TA_model1_1 = ConvNetMaker(plane_cifar100_book.get("8")).to(device)
    TA_model1_2 = ConvNetMaker(plane_cifar100_book.get("8")).to(device)
    TA_model2_1 = ConvNetMaker(plane_cifar100_book.get("6")).to(device)
    TA_model2_2 = ConvNetMaker(plane_cifar100_book.get("6")).to(device)
    TA_model3_1 = ConvNetMaker(plane_cifar100_book.get("4")).to(device)
    TA_model3_2 = ConvNetMaker(plane_cifar100_book.get("4")).to(device)
    co_teaching_student_model = ConvNetMaker(plane_cifar100_book.get("2")).to(device)
    start_time = time.time()
    CTKD_teacher_to_TA(
        teacher=teacher_model,
        TA1=TA_model1_1,
        TA2=TA_model1_2,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        soft_target_loss_weight=0.5,
        ce_loss_weight=0.5,
        device=device,
    )
    TA1_test_accuracy1 = test(TA_model1_1, test_loader, device)
    TA1_test_accuracy2 = test(TA_model1_2, test_loader, device)

    CTKD_TA_to_TA(
        TA_Teacher1=TA_model1_1,
        TA_Teacher2=TA_model1_2,
        TA1=TA_model2_1,
        TA2=TA_model2_2,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        soft_target_loss_weight=0.5,
        ce_loss_weight=0.5,
        device=device,
        remember_rate=1,
    )
    TA2_test_accuracy1 = test(TA_model2_1, test_loader, device)
    TA2_test_accuracy2 = test(TA_model2_2, test_loader, device)

    CTKD_TA_to_TA(
        TA_Teacher1=TA_model2_1,
        TA_Teacher2=TA_model2_2,
        TA1=TA_model3_1,
        TA2=TA_model3_2,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        soft_target_loss_weight=0.5,
        ce_loss_weight=0.5,
        device=device,
        remember_rate=1,
    )
    TA3_test_accuracy1 = test(TA_model3_1, test_loader, device)
    TA3_test_accuracy2 = test(TA_model3_2, test_loader, device)

    CTKD_TA_to_Student(
        TA1=TA_model3_1,
        TA2=TA_model3_2,
        student=co_teaching_student_model,
        train_loader=train_loader,
        epochs=160,
        learning_rate=0.1,
        T=2,
        soft_target_loss_weight=0.5,
        ce_loss_weight=0.5,
        device=device,
        remember_rate=1,
    )
    co_teaching_test_accuracy = test(co_teaching_student_model, test_loader, device)
    end_time = time.time()

    CTKD_time = end_time - start_time"""
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    # print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
    # print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
    print(f"Student accuracy with CE + TAKD: {test_accuracy_light_ce_and_takd:.2f}%")
    print(f"Student accuracy with CE + DGKD: {test_accuracy_light_ce_and_DGKD:.2f}%")
    print(f"co-teaching: {co_teaching_test_accuracy:.2f}%")
