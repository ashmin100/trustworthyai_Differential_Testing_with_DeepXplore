import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

# 재현성 보장을 위한 시드 고정
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def get_resnet50():
    # torchvision의 ResNet50 모델을 준비하여 CIFAR-10용으로 변형합니다
    model = torchvision.models.resnet50(pretrained=False)
    # 32x32 크기의 이미지를 입력으로 받기 위해 첫 번째 컨볼루션 계층(conv1)과 풀링(maxpool) 구조를 축소/변경합니다.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 10개 클래스 출력을 위해 FC 레이어 크기 조절
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_model(model, trainloader, device, epochs=1, name="model"):
    print(f"{name} 를 {epochs} 에포크 동안 학습합니다...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{name}] Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}")
                running_loss = 0.0
    print(f"{name} 학습 완료")
    torch.save(model.state_dict(), f"{name}.pt")
