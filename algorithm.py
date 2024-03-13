import torch
from torch import nn, optim
import random
import torch.nn.functional as F


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
    TA,
    student,
    train_loader,
    epochs,
    learning_rate,
    T,
    lambda_,
    device,
):
    optimizer1 = optim.SGD(
        TA.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )
    optimizer2 = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )

    teacher.eval()  # Teacher set to evaluation mode
    TA.train()
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)
                teacher_soft_labels = torch.softmax(teacher_logits / T, dim=1)

            # Forward pass with the student model
            TA_logits = TA(inputs)
            student_logits = student(inputs)

            student_label_loss = F.cross_entropy(student_logits, labels)
            student_distill_loss = F.kl_div(
                torch.log_softmax(student_logits / T, dim=1),
                teacher_soft_labels,
            )

            TA_label_loss = F.cross_entropy(TA_logits, labels)
            TA_distill_loss = F.kl_div(
                torch.log_softmax(TA_logits / T, dim=1),
                teacher_soft_labels,
            )

            loss = lambda_ * (student_distill_loss + TA_distill_loss) + (
                1 - lambda_
            ) * (student_label_loss + TA_label_loss)

            loss.backward()
            optimizer1.step()
            optimizer2.step()

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
            distill_loss1 = F.kl_div(
                torch.log_softmax(TA1_outputs / T, dim=1),
                teacher_soft_labels,
                reduction="none",
            )
            print(distill_loss1.mean(dim=1))
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
