from matplotlib import cm
import numpy as np
from sklearn.manifold import TSNE
import torch
from torch import nn, optim
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from collections import defaultdict
from ema import EMA


def train(
    model,
    train_loader,
    epochs,
    learning_rate,
    device,
    test_loader,
    name,
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
    loss_list = []
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
        loss_list.append(running_loss / len(train_loader))
        val_acc = test(model, test_loader, device)
        val_acc_list.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"model_list/{name}.pth")

    print_plt(val_acc_list, epochs, f"model_png/{name}", loss_list=loss_list)


def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def test_ensemble(models, test_loader, device):
    criterion = nn.CrossEntropyLoss(reduction="none")
    models = [model.to(device).eval() for model in models]

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = torch.zeros(inputs.size(0), 100).to(device)
            for model in models:
                outputs += model(inputs)
            outputs /= len(models)
            print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def test_ensemble_with_confidence_weighted_average(models, test_loader, device):
    criterion = nn.CrossEntropyLoss(reduction="none")
    models = [model.to(device).eval() for model in models]

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 각 모델의 출력을 저장할 리스트
            all_outputs = []

            for model in models:
                outputs = model(inputs)
                all_outputs.append(outputs)

            # 각 모델의 확신도 계산 (가장 높은 softmax 확률 값)
            confidences = []
            for outputs in all_outputs:
                softmax_outputs = nn.functional.softmax(outputs, dim=1)
                max_confidence, _ = torch.max(softmax_outputs, dim=1)
                confidences.append(max_confidence.mean().item())

            # 확신도를 가중치로 사용하여 가중 평균 계산 (확신도를 제곱하여 차이를 두드러지게 함)
            confidences = torch.tensor(confidences, device=device)
            confidences = torch.exp(confidences)  # 확신도를 제곱하여 차이를 확대
            weights = confidences / confidences.sum()

            print(weights)
            # 가중 평균 계산
            weighted_outputs = torch.zeros(inputs.size(0), 100).to(device)
            for i, outputs in enumerate(all_outputs):
                weighted_outputs += weights[i] * outputs

            print(weighted_outputs.shape)
            _, predicted = torch.max(weighted_outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def test_ensemble_class_accuracy(models, test_loader, device):
    models = [model.to(device).eval() for model in models]

    # 클래스별로 맞춘 개수와 전체 개수를 저장할 딕셔너리
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 각 모델의 출력을 저장할 리스트
            all_outputs = []

            for model in models:
                outputs = model(inputs)
                all_outputs.append(outputs)

            # 각 모델의 확신도 계산 (가장 높은 softmax 확률 값)
            confidences = []
            for outputs in all_outputs:
                softmax_outputs = nn.functional.softmax(outputs, dim=1)
                max_confidence, _ = torch.max(softmax_outputs, dim=1)
                confidences.append(max_confidence.mean().item())

            # 확신도를 가중치로 사용하여 가중 평균 계산
            confidences = torch.tensor(confidences, device=device)
            confidences = confidences**2  # 확신도를 제곱하여 차이를 확대
            weights = confidences / confidences.sum()

            # 가중 평균 계산
            weighted_outputs = torch.zeros(inputs.size(0), 100).to(device)
            for i, outputs in enumerate(all_outputs):
                weighted_outputs += weights[i] * outputs

            _, predicted = torch.max(weighted_outputs.data, 1)

            # 각 클래스에 대한 정확도 계산
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    # 클래스별 정확도 계산
    class_accuracy = {
        cls: 100 * class_correct[cls] / class_total[cls] for cls in class_total
    }

    # 정확도에 따라 클래스별로 정렬
    sorted_class_accuracy = sorted(class_accuracy.items(), key=lambda item: item[0])

    probabilities = [prob for _, prob in sorted_class_accuracy]

    return probabilities


def test_for_error(model, test_loader, device):
    model.to(device)
    model.eval()

    label_accuracy = []  # 라벨별 정답 여부를 저장할 배열

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            print(labels)
            # 라벨별 정답 여부를 배열에 추가
            label_accuracy.extend((predicted != labels).cpu().tolist())

    print(label_accuracy)
    return np.array(label_accuracy)


def error_overlap_loss(y_true, y_pred_teacher, y_pred_student):
    teacher_predictions = torch.argmax(y_pred_teacher, dim=1)
    student_predictions = torch.argmax(y_pred_student, dim=1)
    true_max = torch.argmax(y_true, dim=1)

    # 예측이 틀렸는지 확인
    teacher_errors = teacher_predictions != true_max
    student_errors = student_predictions != true_max

    overlap_errors = teacher_errors & student_errors
    same_error_class = teacher_predictions == student_predictions
    error_overlap_loss = (
        torch.logical_and(overlap_errors, same_error_class).float().mean()
    )

    # 오버랩된 틀린 예측에 대해 손실 부여
    return error_overlap_loss


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


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
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    teacher.eval()  # Teacher set to evaluation mode
    val_acc_list = []
    loss_list = []
    best_acc = 0

    ce_loss_list = []
    kl_loss_list = []
    for epoch in range(epochs):
        # cifar_class_per_value = torch.zeros(100, device=device)
        # cifar_class_count = torch.zeros(100, device=device)

        # cifar_distill_per_value = torch.zeros(100, device=device)
        # cifar_distill_count = torch.zeros(100, device=device)

        student.train()  # Student to train mode
        running_loss = 0.0

        for inputs, labels, indices in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_soft_labels = torch.softmax(teacher_logits / T, dim=1)

            # Forward pass with the student model
            student_logits = student(inputs)

            student_label_loss = criterion(student_logits, labels)

            # for label, loss in zip(labels, student_label_loss):
            #     cifar_class_count[label] += 1
            #     cifar_class_per_value[label] += loss

            student_label_loss = student_label_loss.mean()
            distill_loss = nn.KLDivLoss(reduction="none")(
                torch.log_softmax(student_logits / T, dim=1),
                teacher_soft_labels,
            )

            # for label, loss in zip(labels, distill_loss.mean(dim=1)):
            #     cifar_distill_count[label] += 1
            #     cifar_distill_per_value[label] += loss

            distill_loss = distill_loss.mean()
            loss = lambda_ * student_label_loss + (1 - lambda_) * T * T * distill_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # cifar_class_per_value /= cifar_class_count.clamp(min=1)
        # print(cifar_class_per_value)

        # cifar_class_per_value_cpu = (
        #     cifar_class_per_value.detach().cpu().numpy().reshape(10, 10)
        # )

        # print_cifar_heatmap(
        #     cifar_class_per_value_cpu,
        #     epoch,
        #     running_loss / len(train_loader),
        # )

        # cifar_distill_per_value /= cifar_distill_count.clamp(min=1)
        # print(cifar_distill_per_value)

        # cifar_distill_per_value_cpu = (
        #     cifar_distill_per_value.detach().cpu().numpy().reshape(10, 10)
        # )
        # print_cifar_distill_heatmap(
        #     cifar_distill_per_value_cpu,
        #     epoch,
        #     running_loss / len(train_loader),
        # )

        loss_list.append(running_loss / len(train_loader))
        val_acc = test(student, test_loader, device)
        val_acc_list.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), f"model_list/TAKD/{name}_T_{T}.pth")

    return val_acc_list
    # print_plt(val_acc_list, epochs, f"model_png/TAKD/{name}_T_{T}", loss_list=loss_list)


def test_kd_model(
    student,
    teacher_list,
    class_table,
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

    val_acc_list = []
    loss_list = []
    best_acc = 0

    teacher_list = [model.eval() for model in teacher_list]
    class_table = [torch.tensor(table).to(device) for table in class_table]

    for epoch in range(epochs):
        start_time = time.time()

        student.train()  # Student to train mode
        running_loss = 0.0
        for inputs, labels, indices in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            teacher_avg_output = torch.zeros(inputs.size(0), 100).to(device)

            student_logits = student(inputs)
            loss_SL = F.cross_entropy(student_logits, labels)

            with torch.no_grad():
                all_outputs = []

                for teacher, table in zip(teacher_list, class_table):
                    outputs = teacher(inputs)
                    outputs_softmax = torch.softmax(outputs, reduction="none")
                    print(outputs_softmax.shape, table.shape)
                    outputs_softmax *= table[indices]
                    all_outputs.append(outputs)

                teacher_avg_output = sum(all_outputs) / len(teacher_list)

            loss_KD = nn.KLDivLoss()(
                F.log_softmax(student_logits / T, dim=1), teacher_avg_output
            )

            loss = lambda_ * loss_SL + (1 - lambda_) * T * T * loss_KD

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print((time.time() - start_time))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        loss_list.append(running_loss / len(train_loader))
        val_acc = test(student, test_loader, device)
        val_acc_list.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), f"model_list/test/{name}_T_{T}.pth")

    print_plt(val_acc_list, epochs, f"model_png/test/{name}_T_{T}", loss_list=loss_list)


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
    loss_list = []
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
        loss_list.append(running_loss / len(train_loader))
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
    loss_list = []
    best_acc = 0

    for epoch in range(epochs):
        student.train()  # Student to train mode
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
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
                nn.KLDivLoss()(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teahcer_logits / T, dim=1),
                )
            ]

            # Teacher Assistants Knowledge Distillation Loss
            for i in range(len(ta_list)):
                loss_KD_list.append(
                    nn.KLDivLoss()(
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
        loss_list.append(running_loss / len(train_loader))
        val_acc = test(student, test_loader, device)
        val_acc_list.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), f"model_list/DGKD/{name}_T_{T}.pth")

    print_plt(val_acc_list, epochs, f"model_png/DGKD/{name}_T_{T}", loss_list=loss_list)


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
    loss_list=None,
):
    plt.clf()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(
        range(1, epochs + 1), val_acc_list, marker="o", color="b", label="Test Accuracy"
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if loss_list:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Loss", color="tab:red")
        ax2.plot(
            range(1, epochs + 1), loss_list, marker="o", color="r", label="Train Loss"
        )
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_yscale("log")

    fig.tight_layout()

    plt.title(f"{name}Test Accuracy by Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}.png")
    plt.close()


def print_double_plt(
    val_acc_list1,
    val_acc_list2,
    epochs,
    name,
):
    plt.clf()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(
        range(1, epochs + 1),
        val_acc_list1,
        marker="o",
        color="b",
        label="default",
    )
    ax1.plot(range(1, epochs + 1), val_acc_list2, marker="x", color="r", label="KD")

    ax1.tick_params(axis="y", labelcolor="tab:blue")
    fig.legend(loc="upper right")  # 범례 위치 조정
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f"{name}.png")
    plt.close()


def print_cifar_heatmap(tensor, epoch, loss):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        tensor, cmap="viridis", norm=colors.LogNorm(vmin=1e-3, vmax=5)
    )  # cmap 파라미터로 원하는 컬러맵 지정 가능
    plt.colorbar()  # 색상 막대 표시

    plt.savefig(f"heatmap/epoch {epoch} loss {loss}.png")
    plt.close()


def print_cifar_distill_heatmap(tensor, epoch, loss):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        tensor, cmap="viridis", norm=colors.LogNorm(vmin=1e-4, vmax=1e-1)
    )  # cmap 파라미터로 원하는 컬러맵 지정 가능
    plt.colorbar()  # 색상 막대 표시

    plt.savefig(f"distill_heatmap/epoch {epoch} loss {loss}.png")
    plt.close()


def get_one_heatmap(model, device, test_loader):
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    cifar_class_per_value = torch.zeros(100, device=device)
    cifar_class_count = torch.zeros(100, device=device)

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(inputs)
            label_loss = criterion(logits, labels)

        for label, loss in zip(labels, label_loss):
            cifar_class_count[label] += 1
            cifar_class_per_value[label] += loss

    cifar_class_per_value /= cifar_class_count.clamp(min=1)

    cifar_class_per_value_cpu = (
        cifar_class_per_value.detach().cpu().numpy().reshape(10, 10)
    )
    print_cifar_heatmap(cifar_class_per_value_cpu, -1, label_loss.mean())
    plt.close()


def print_weight_data(weight, epoch):
    weight = weight.cpu().numpy()

    print(weight)
    plt.figure(figsize=(10, 8))
    plt.imshow(
        weight, cmap="viridis", vmin=-1, vmax=1
    )  # cmap 파라미터로 원하는 컬러맵 지정 가능
    plt.colorbar()  # 색상 막대 표시

    plt.savefig(f"weight/weight_epoch_{epoch}.png")
    plt.close()


def create_model_table(data_loader, model, T, device):
    model_table = torch.zeros((50000, 100), device=device)
    model.eval()
    for input, _, idx in data_loader:
        with torch.no_grad():
            input = input.to(device)

            logits = model(input)
            soft_logits = F.softmax(logits / T, dim=1)
            model_table[idx] = soft_logits

    return model_table
