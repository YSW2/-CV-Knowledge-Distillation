from matplotlib import cm
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch import nn, optim
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train(
    model,
    train_loader,
    epochs,
    learning_rate,
    device,
    test_loader,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    model.train()
    val_acc_list = []
    best_acc = 0

    for epoch in range(epochs):
        model.train()
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
        val_acc = test(model, test_loader, device)
        val_acc_list.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"model_list/teacher_model.pth")

    print_plt(val_acc_list, epochs, f"model_png/teacher_model")


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


def test_for_error(model, test_loader, device):
    model.to(device)
    model.eval()

    label_accuracy = []  # 라벨별 정답 여부를 저장할 배열

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # 라벨별 정답 여부를 배열에 추가
            label_accuracy.extend((predicted == labels).cpu().tolist())

    return np.array(label_accuracy)


def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    epochs,
    learning_rate,
    T,
    lambda_,
    device,
    test_loader=None,
    name=None,
):

    optimizer = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    teacher.eval()  # Teacher set to evaluation mode
    val_acc_list = []
    best_acc = 0

    for epoch in range(epochs):
        student.train()  # Student to train mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_soft_labels = torch.softmax(teacher_logits / T, dim=1)

            # Forward pass with the student model
            student_logits = student(inputs)

            student_label_loss = F.cross_entropy(student_logits, labels)
            distill_loss = F.kl_div(
                torch.log_softmax(student_logits / T, dim=1),
                teacher_soft_labels,
            )

            loss = lambda_ * student_label_loss + (1 - lambda_) * T * T * distill_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        val_acc = test(student, test_loader, device)
        val_acc_list.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), f"model_list/TAKD/{name}_T_{T}.pth")

    print_plt(val_acc_list, epochs, f"model_png/TAKD/{name}_T_{T}")


def CTKD_teacher_to_TA(
    teacher,
    TA1,
    TA2,
    train_loader,
    epochs,
    learning_rate,
    T,
    lambda_,
    device,
    test_loader=None,
    name=None,
):
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

    best_acc_TA1 = 0
    best_acc_TA2 = 0

    val_acc_list1 = []
    val_acc_list2 = []

    teacher.eval()  # Teacher set to evaluation mode

    for epoch in range(epochs):
        TA1.train()  # Student to train mode
        TA2.train()
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
            distill_loss1 = F.kl_div(
                torch.log_softmax(TA1_outputs / T, dim=1),
                teacher_soft_labels,
            )

            total_loss1 = lambda_ * loss1 + (1 - lambda_) * T * T * distill_loss1
            total_loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2 = F.cross_entropy(TA2_outputs, labels)
            distill_loss2 = F.kl_div(
                torch.log_softmax(TA2_outputs / T, dim=1),
                teacher_soft_labels,
            )
            total_loss2 = lambda_ * loss2 + (1 - lambda_) * T * T * distill_loss2
            total_loss2.backward()
            optimizer2.step()

            running_loss += total_loss1.item() + total_loss2.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        val_acc_TA1 = test(TA1, test_loader, device)
        val_acc_TA2 = test(TA2, test_loader, device)
        val_acc_list1.append(val_acc_TA1)
        val_acc_list2.append(val_acc_TA2)
        if val_acc_TA1 > best_acc_TA1:
            best_acc_TA1 = val_acc_TA1
            torch.save(TA1.state_dict(), f"model_list/test/{name}_1_T_{T}.pth")
        if val_acc_TA2 > best_acc_TA2:
            best_acc_TA2 = val_acc_TA2
            torch.save(TA2.state_dict(), f"model_list/test/{name}_2_T_{T}.pth")

    print_plt(val_acc_list1, epochs, f"model_png/test/{name}_1_T_{T}")
    print_plt(val_acc_list2, epochs, f"model_png/test/{name}_2_T_{T}")


def CTKD_TA_to_TA(
    TA_Teacher1,
    TA_Teacher2,
    TA1,
    TA2,
    train_loader,
    epochs,
    learning_rate,
    T,
    lambda_,
    device,
    remember_rate,
    test_loader=None,
    name=None,
):
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

    best_acc_TA1 = 0
    best_acc_TA2 = 0

    val_acc_list1 = []
    val_acc_list2 = []

    TA_Teacher1.eval()  # Teacher set to evaluation mode
    TA_Teacher2.eval()

    for epoch in range(epochs):
        TA1.train()  # Student to train mode
        TA2.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                TA_Teacher1_logits = TA_Teacher1(inputs)
                TA_Teacher2_logits = TA_Teacher2(inputs)

                TA_Teacher1_soft_labels = torch.softmax(TA_Teacher1_logits / T, dim=1)
                TA_Teacher2_soft_labels = torch.softmax(TA_Teacher2_logits / T, dim=1)

            TA1_outputs = TA1(inputs)
            TA2_outputs = TA2(inputs)

            optimizer1.zero_grad()
            loss1 = F.cross_entropy(TA1_outputs, labels)
            distill_loss1 = F.kl_div(
                torch.log_softmax(TA1_outputs / T, dim=1),
                TA_Teacher1_soft_labels,
            )
            total_loss1 = lambda_ * loss1 + (1 - lambda_) * T * T * distill_loss1
            total_loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss2 = F.cross_entropy(TA2_outputs, labels)
            distill_loss2 = F.kl_div(
                torch.log_softmax(TA2_outputs / T, dim=1),
                TA_Teacher2_soft_labels,
            )
            total_loss2 = lambda_ * loss2 + (1 - lambda_) * T * T * distill_loss2
            total_loss2.backward()
            optimizer2.step()

            running_loss += total_loss1.item() + total_loss2.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        val_acc_TA1 = test(TA1, test_loader, device)
        val_acc_TA2 = test(TA2, test_loader, device)
        val_acc_list1.append(val_acc_TA1)
        val_acc_list2.append(val_acc_TA2)
        if val_acc_TA1 > best_acc_TA1:
            best_acc_TA1 = val_acc_TA1
            torch.save(TA1.state_dict(), f"model_list/test/{name}_1_T_{T}.pth")
        if val_acc_TA2 > best_acc_TA2:
            best_acc_TA2 = val_acc_TA2
            torch.save(TA2.state_dict(), f"model_list/test/{name}_2_T_{T}.pth")

    print_plt(val_acc_list1, epochs, f"model_png/test/{name}_1_T_{T}")
    print_plt(val_acc_list2, epochs, f"model_png/test/{name}_2_T_{T}")


def CTKD_TA_to_Student(
    TA1,
    TA2,
    student,
    train_loader,
    epochs,
    learning_rate,
    T,
    lambda_,
    device,
    remember_rate,
    test_loader=None,
    name=None,
):
    optimizer = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    TA1.eval()  # Teacher set to evaluation mode
    TA2.eval()
    val_acc_list = []
    best_acc = 0

    for epoch in range(epochs):
        student.train()  # Student to train mode
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
            selected_inputs1 = inputs[topk_indices1]
            selected_labels1 = labels[topk_indices1]
            selected_inputs2 = inputs[topk_indices2]
            selected_labels2 = labels[topk_indices2]

            student1_outputs = student(selected_inputs1)
            student2_outputs = student(selected_inputs2)

            optimizer.zero_grad()
            loss1 = F.cross_entropy(student1_outputs, selected_labels1)
            distill_loss1 = F.kl_div(
                torch.log_softmax(student1_outputs / T, dim=1),
                TA2_soft_labels[topk_indices2],
            )
            total_loss1 = lambda_ * loss1 + (1 - lambda_) * T * T * distill_loss1
            total_loss1.backward()

            loss2 = F.cross_entropy(student2_outputs, selected_labels2)
            distill_loss2 = F.kl_div(
                torch.log_softmax(student2_outputs / T, dim=1),
                TA1_soft_labels[topk_indices1],
            )
            total_loss2 = lambda_ * loss2 + (1 - lambda_) * T * T * distill_loss2
            total_loss2.backward()
            optimizer.step()

            running_loss += total_loss1.item() + total_loss2.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        val_acc = test(student, test_loader, device)
        val_acc_list.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), f"model_list/test/{name}_T_{T}.pth")

    print_plt(val_acc_list, epochs, f"model_png/test/{name}_T_{T}")


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
    test_loader=None,
    name=None,
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
    val_acc_list = []
    best_acc = 0

    for epoch in range(epochs):
        student.train()  # Student to train mode
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

            loss = lambda_ * loss_SL + (1 - lambda_) * T * T * sum(loss_KD_list)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        val_acc = test(student, test_loader, device)
        val_acc_list.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), f"model_list/DGKD/{name}_T_{T}.pth")

    print_plt(val_acc_list, epochs, f"model_png/DGKD/{name}_T_{T}")


def print_tSNE(model, dataloader, device):
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


def print_plt(
    val_acc_list,
    epochs,
    name,
):
    plt.clf()
    plt.plot(
        range(1, epochs + 1), val_acc_list, marker="o", color="b", label="Test Accuracy"
    )
    plt.title(f"{name}Test Accuracy by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}.png")


def print_error_rate(model_list, test_loader, device, name):
    error_list = []
    overlap_errors_list = []

    for model in model_list:
        error_list.append(test_for_error(model, test_loader, device))

    for i in range(1, len(error_list)):
        overlap_errors = np.logical_not(
            (np.logical_or(error_list[i - 1], error_list[i]))
        )
        total_errors = np.logical_not(
            (np.logical_and(error_list[i - 1], error_list[i]))
        )
        overlap_errors_list.append(np.sum((overlap_errors) / np.sum(total_errors)))

    plt.clf()
    plt.bar(
        [f"model{i} & model1{i+1}" for i in range(len(overlap_errors_list))],
        overlap_errors_list,
        color=["blue"],
    )
    plt.title(f"{name} error overlap rate")
    plt.ylabel("Rate (%)")
    plt.ylim(0, 1)
    plt.savefig(f"error_overlap_rate/{name}_error_overlap.png")
