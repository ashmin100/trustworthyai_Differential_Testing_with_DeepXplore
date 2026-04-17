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

# DeepXplore 구현을 위한 보조 함수
def deprocess_image(tensor):
    # 역정규화를 진행하여 이미지 시각화가 가능하도록 복원합니다.
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
    STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    img = tensor * STD + MEAN
    img = torch.clamp(img, 0, 1)
    return img.cpu().detach().permute(1, 2, 0).numpy()

class ModelWithHooks:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        # 전체 신경망 커버리지를 측정하기 위해 모든 ReLU 레이어에 Hook을 부착합니다.
        self._register_hooks(self.model, prefix="")
        
    def _register_hooks(self, module, prefix):
        for name, child in module.named_children():
            full_name = f"{prefix}_{name}" if prefix else name
            if isinstance(child, nn.ReLU):
                child.register_forward_hook(self._get_activation(full_name))
            else:
                self._register_hooks(child, full_name)
        
    def _get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output
        return hook
        
    def __call__(self, x):
        return self.model(x)

def deepxplore_generation(model1_wrapper, model2_wrapper, seed_img, orig_label, threshold=0.1, weight_diff=1.0, weight_nc=0.1, step_size=0.01, iters=50, device='cpu'):
    # DeepXplore의 목표: 뉴런 커버리지를 최대화하면서 기투자된 모델 두 개 간의 예측값 차이를 유도하는 것
    img = seed_img.clone().detach().to(device)
    img.requires_grad_(True)
    
    # DeepXplore (SOSP 2017) 논문 참고   
    # 뉴런 커버리지는 소프트웨어 테스트의 statement coverage와 대응되는 개념입니다.
    # 특정 뉴런의 활성값(ReLU 거친 값 vk,i)이 0보다 크면(threshold=0) 해당 뉴런이 '커버(covered)'되었다고 간주합니다.
    threshold = 0.0 # uk,i > 0 이면 커버된 것.
    
    covered_neurons_m1 = set()
    covered_neurons_m2 = set()
    
    for i in range(iters):
        out1 = model1_wrapper(img)
        out2 = model2_wrapper(img)
        
        pred1 = out1.argmax(dim=1).item()
        pred2 = out2.argmax(dim=1).item()
        
        # NC(T) = |Cov(T)| / 총 뉴런 수 
        # CNN의 경우 공간 차원(H, W) 평균을 구한 뒤 활성 여부(> 0)를 판단합니다.
        for layer_name, act1 in model1_wrapper.activations.items():
            if act1.dim() == 4: # 컨볼루션 계층 (B, C, H, W)
                active1 = (act1.mean(dim=(2,3)) > threshold).nonzero()
                for idx in active1:
                    covered_neurons_m1.add(f"{layer_name}_{idx[1].item()}")
            
        for layer_name, act2 in model2_wrapper.activations.items():
            if act2.dim() == 4:
                active2 = (act2.mean(dim=(2,3)) > threshold).nonzero()
                for idx in active2:
                    covered_neurons_m2.add(f"{layer_name}_{idx[1].item()}")

        print(f"반복(Iter) {i}: 예측1={pred1}, 예측2={pred2}, 커버리지 1={len(covered_neurons_m1)}, 커버리지 2={len(covered_neurons_m2)}")

        if pred1 != pred2:
            # 두 모델 사이에 불일치(Discrepancy) 시점 발견 조기 종료!
            return img, pred1, pred2, len(covered_neurons_m1), len(covered_neurons_m2), True
            
        # 미활성화 뉴런을 활성화하도록 모든 계층의 활성도 총합을 목적함수에 반영합니다.
        loss1_neuron = sum(act.mean() for act in model1_wrapper.activations.values())
        loss2_neuron = sum(act.mean() for act in model2_wrapper.activations.values())
        
        # 주 목적함수: model2는 여전히 원본 정답을 예측하게 하면서, 
        # model1의 예측값은 정답이 아니게 되도록 (음수로) 손실 함수를 형성합니다.
        loss_diff = out2[0, orig_label] - out1[0, orig_label] * weight_diff
        
        # 최종적으로 커버리지 가중치를 더해줍니다.
        loss = loss_diff + weight_nc * (loss1_neuron + loss2_neuron)
        
        # 손실 함수를 최대화하는 방향으로 기울기 계산 (Gradient Ascent)
        model1_wrapper.model.zero_grad()
        model2_wrapper.model.zero_grad()
        if img.grad is not None:
            img.grad.zero_()
        
        loss.backward()
        
        # 경사 상승법(Gradient Ascent) 적용 - 입력 이미지 업데이트
        with torch.no_grad():
            img += step_size * torch.sign(img.grad) # 보다 빠르고 강한 생성을 위해 기울기 부호(Sign) 활용
            
        img.requires_grad_(True)
        
    return img, pred1, pred2, len(covered_neurons_m1), len(covered_neurons_m2), False



def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치(Device): {device}")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("CIFAR-10 데이터셋 로드 중...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # 10개 클래스 레이블 매핑
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 두 개의 ResNet50 모델 생성
    model1 = get_resnet50().to(device)
    model2 = get_resnet50().to(device)
    
    if os.path.exists("model1.pt") and os.path.exists("model2.pt"):
        print("기존에 학습된 모델(model1.pt, model2.pt)을 불러옵니다...")
        model1.load_state_dict(torch.load("model1.pt", map_location=device))
        model2.load_state_dict(torch.load("model2.pt", map_location=device))
    else:
        print("초기화 가중치가 서로 다른 모델 두 개를 생성하여 각 1 에포크씩 훈련합니다...")
        # 두 모델이 서로 의미 있는 차이를 발생시키도록 시드를 설정합니다
        torch.manual_seed(10)
        model1 = get_resnet50().to(device)
        torch.manual_seed(20)
        model2 = get_resnet50().to(device)
        # 매우 적은 훈련만 거쳐도 테스트와 불일치 생성이 가능해집니다.
        train_model(model1, trainloader, device, epochs=1, name="model1")
        train_model(model2, trainloader, device, epochs=1, name="model2")
        
    model1.eval()
    model2.eval()
    
    # 모델에 중간 레이어 Hook 을 부착하기 위한 Wrapper 클래스 생성
    model1_wrapper = ModelWithHooks(model1)
    model2_wrapper = ModelWithHooks(model2)

    os.makedirs("results", exist_ok=True)
    
    discrepancies_found = 0
    target_discrepancies = 5
    
    # 커버리지 변화량을 추적하여 차트로 그리기 위한 리스트 초기화
    m1_cov_history = []
    m2_cov_history = []
    
    print("DeepXplore 차이점 테스트(Differential Testing) 생성을 시작합니다...")
    # 테스트 셋 이미지들을 무작위 순서로 섞어서 시드(seed) 이미지로 사용
    indices = torch.randperm(len(testset))
    
    for idx in indices:
        seed_img, orig_label = testset[idx]
        seed_img = seed_img.unsqueeze(0).to(device)
        
        # 노이즈를 섞기 전 초기 예측 결과 확인
        out1 = model1(seed_img)
        out2 = model2(seed_img)
        pred1 = out1.argmax(dim=1).item()
        pred2 = out2.argmax(dim=1).item()
        
        if pred1 != pred2:
            print(f"해당 이미지 번호 {idx} 에서는 두 모델이 이미 다른 예측을 하고 있습니다. (M1: {classes[pred1]}, M2: {classes[pred2]}). 스킵합니다.")
            continue
            
        print(f"\n시드 이미지 평가 중 - 인덱스 번호 {idx} (실제 정답 레이블: {classes[orig_label]})")
        
        # DeepXplore 이미지 생성 진행
        gen_img, fpred1, fpred2, cov1, cov2, success = deepxplore_generation(
            model1_wrapper, model2_wrapper, seed_img, orig_label, 
            threshold=0.1, weight_diff=0.5, weight_nc=0.1, step_size=0.01, iters=30, device=device
        )
        
        if success:
            discrepancies_found += 1
            # 동적으로 추적된 전체 채널(뉴런) 수 계산
            total_neurons = sum(act.shape[1] for act in model1_wrapper.activations.values() if act.dim() == 4)
            cov1_pct = (cov1 / total_neurons) * 100
            cov2_pct = (cov2 / total_neurons) * 100
            
            print(f"성공! 모델 예측 불일치 이미지를 생성했습니다. M1 예측: {classes[fpred1]}, M2 예측: {classes[fpred2]}")
            print(f"Chapter 5 기준: 전체 뉴런(ReLU) 커버리지 도달 확인 - M1: {cov1}개 ({cov1_pct:.2f}%), M2: {cov2}개 ({cov2_pct:.2f}%)")
            
            # 시각화 후 저장
            orig_vis = deprocess_image(seed_img[0])
            gen_vis = deprocess_image(gen_img[0])
            noise_vis = gen_vis - orig_vis
            # 노이즈 변경점이 명확히 보이도록 동적 스케일링
            noise_vis = (noise_vis - np.min(noise_vis)) / (np.max(noise_vis) - np.min(noise_vis) + 1e-5)
            
            # 시각화 해상도 및 퀄리티 업그레이드 
            # figsize를 키우고, 폰트 크기를 늘리며, 고해상도 저장을 위해 dpi를 적용합니다.
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # CIFAR-10 32x32 미니 이미지가 강제로 늘어날 때 뿌옇게 흐려지는 것을 방지(픽셀 선명도 유지)
            ax[0].imshow(orig_vis, interpolation='nearest')
            ax[0].set_title(f"Original (Pred: {classes[pred1]})", fontsize=14, pad=10)
            ax[0].axis('off')
            
            ax[1].imshow(gen_vis, interpolation='nearest')
            ax[1].set_title(f"Perturbed (M1:{classes[fpred1]} M2:{classes[fpred2]})", fontsize=14, pad=10)
            ax[1].axis('off')
            
            ax[2].imshow(noise_vis, interpolation='nearest')
            ax[2].set_title("Perturbation Noise", fontsize=14, pad=10)
            ax[2].axis('off')
            
            plt.tight_layout()
            # dpi=300 옵션으로 학술 논문 수준의 고화질 이미지(PNG)로 저장합니다.
            plt.savefig(f"results/disagreement_{discrepancies_found}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 성공한 이미지에 대한 커버리지 기록 (백분율 값 저장)
            m1_cov_history.append(cov1_pct)
            m2_cov_history.append(cov2_pct)
            
            if discrepancies_found >= target_discrepancies:
                print(f"목표한 {target_discrepancies}개의 모델 불일치 결과 이미지를 성공적으로 찾았습니다.")
                
                # 최종 커버리지 비율 차트(Bar Chart) 시각화 및 저장
                print("최종 커버리지 비율 요약 차트를 생성 중입니다...")
                plt.figure(figsize=(10, 6), dpi=300)
                indices_arr = np.arange(1, target_discrepancies + 1)
                bar_width = 0.35
                
                bars1 = plt.bar(indices_arr - bar_width/2, m1_cov_history, width=bar_width, label='Model 1 (%)', color='#4C72B0')
                bars2 = plt.bar(indices_arr + bar_width/2, m2_cov_history, width=bar_width, label='Model 2 (%)', color='#DD8452')
                
                # 막대 그래프 위에 정확한 % 수치 텍스트 부착
                for bar in bars1:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)
                for bar in bars2:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)
                
                plt.title("Neuron Coverage Percentage per Disagreement Image", fontsize=16)
                plt.xlabel("Generated Adversarial Image Index", fontsize=14)
                plt.ylabel("Neuron Coverage (%)", fontsize=14)
                plt.ylim(0, max(max(m1_cov_history), max(m2_cov_history)) + 10) # y축을 %가 잘 보이도록 조금 더 높게 확보
                plt.xticks(indices_arr, [f"Img {i}" for i in indices_arr])
                plt.legend(fontsize=12, loc='lower right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig("results/coverage_summary_chart.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("차트 생성이 완료되었습니다: results/coverage_summary_chart.png")
                break
                
if __name__ == "__main__":
    main()
