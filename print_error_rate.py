from matplotlib import pyplot as plt
import numpy as np
import torch


def print_error_rate(model_list, test_loader, device, name):
    error_list = []
    overlap_errors_list = []

    for model in model_list:
        model.to(device)
        model.eval()

        label_accuracy = []  # 라벨별 정답 여부를 저장할 배열

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # 라벨별 정답 여부를 배열에 추가
                label_accuracy.extend((predicted != labels).cpu().tolist())

        error_list.append(np.array(label_accuracy))

    for i in range(1, len(error_list)):
        overlap_errors = error_list[i - 1] & error_list[i]
        overlap_errors_list.append(
            overlap_errors.sum().astype(float) / error_list[i - 1].sum() * 100
        )

    plt.clf()
    plt.bar(
        [f"model{i+1} & model1{i+2}" for i in range(len(overlap_errors_list))],
        overlap_errors_list,
        color=["blue"],
    )
    plt.title(f"{name} error overlap rate")
    plt.ylabel("Rate (%)")
    plt.ylim(min(overlap_errors_list) - 3, max(overlap_errors_list) + 3)
    plt.savefig(f"error_overlap_rate/{name}_error_overlap.png")
    plt.close()
