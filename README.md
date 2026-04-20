# DeepXplore를 활용한 차이점 테스트(Differential Testing)

## 실행 설정 가이드

1. 제공된 `requirements.txt`를 사용하여 필요한 종속성 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

2. 테스트 스크립트를 실행합니다:
   ```bash
   python test.py
   ```

## DeepXplore 수정 사항 안내
원래 DeepXplore 코드는 더 이상 잘 쓰이지 않는 오래된 Keras와 TensorFlow 1.x 프레임워크를 기반으로 작성되어 있었습니다. "최신 프레임워크에 맞게 수정(adapt it to work with current frameworks)"하기 위하여, DeepXplore의 핵심 로직 전체를 **PyTorch** 딥러닝 프레임워크에 맞게 전부 재작성(Porting)했습니다. 

- **PyTorch 적용(Adaptation)**: 기존의 `gen_diff.py` 스크립트를 `test.py`로 대체했습니다. `autograd` 및 `register_forward_hook` 함수를 사용하여 중간 레이어 활성화 값을 가로채어 뉴런 커버리지(Neuron Coverage)를 계산하고, 미분을 이용한 경사 상승법(Gradient Ascent)으로 이미지를 변형하는 로직을 PyTorch에 맞게 구현했습니다.
- **CIFAR-10용 ResNet50 모델 구성**: `torchvision.models.resnet50`의 구조(처음 Convolution 계층과 마지막 fully-connected 계층)를 32x32 크기의 CIFAR-10 이미지 및 10개 클래스 출력에 맞추어 변형했습니다. 두 모델이 서로 의미 있는 차이를 보이도록 무작위 시드(Seed)를 각각 да르게 주어 초기화한 뒤(각각 `model1`과 `model2`), 자동으로 1에포크(Epoch) 동안 짧게 학습을 진행하여 최소한의 분류 능력을 갖추도록 만듭니다. (미리 저장된 `model1.pt`, `model2.pt`가 없을 경우 실행 시 자동으로 이를 새로 구성하고 훈련합니다).
- **전체 신경망 뉴런 커버리지 탐색(Total Neuron Coverage)**: 특정 단일 계층만 관찰하는 것이 아니라, 재귀적 알고리즘을 통해 모델 내의 **모든 은닉 활성화 계층(전체 `nn.ReLU`)** 출력에 접근하여 활성화 정도를 감시합니다. 학술적인 엄밀한 테스트와 노이즈 필터링을 위해 지정된 임계값(Threshold, 예: 0.1)을 상회하는 활성 채널 개수를 전부 합산한 뒤, 전체 뉴런 수 대비 도달한 백분율(%)로 명확한 커버리지를 보고합니다.

## 실행 결과
`test.py`를 실행하면 `results/` 폴더 내에 두 모델 간 예측 결과 불일치를 유도한 이미지 결과물이 5개 이상 생성됩니다.
각 이미지는 아래 3가지 요소를 시각적으로 보여줍니다:
1. **원본 이미지 (Original)**
2. **모델 간 예측 차이를 발생시킨 변형 이미지 (Perturbed)**
3. **더해진 노이즈 변화량 (Perturbation Noise)**
4. **커버리지 달성 요약 차트 (`coverage_summary_chart.png`)**: 생성된 5개의 결과 이미지 각각에 대해 두 모델이 얼마만큼의 뉴런 커버리지 백분율(%)을 달성했는지 한눈에 비교할 수 있는 막대 그래프(Bar Chart)입니다.

---

## 코드 동작 원리 및 세부 설명 (How `test.py` works)

`test.py`는 원본 DeepXplore의 핵심 철학인 **"차이점 테스트(Differential Testing) + 뉴런 커버리지(Neuron Coverage) 극대화"**를 PyTorch 환경에서 동작하도록 아래와 같은 파이프라인으로 구현했습니다.

### 1. 전역 후킹(Global Hooking)을 통한 활성화 맵 추출
- `ModelWithHooks` 클래스는 재귀 함수(`_register_hooks`)를 사용하여 모델(ResNet50) 내에 존재하는 **모든 `nn.ReLU` 계층**을 찾아내어 PyTorch의 `register_forward_hook`을 부착합니다. 
- 이 덕분에 추론(Forward pass)이 진행될 때마다 네트워크 전반에서 연산된 수많은 은닉층(Hidden layers) 뉴런들의 활성화 출력값(Activation 텐서)들이 `self.activations` 딕셔너리에 자동으로 수집됩니다.

### 2. 동적 뉴런 커버리지 측정 (Coverage Tracking)
- 수집된 모든 계층의 활성화 텐서들의 공간 차원 평균(Spatial mean)을 구한 뒤, 미리 지정한 임계값(`threshold=0.1`)보다 높을 경우 해당 채널(뉴런)이 **활성화(Covered)** 되었다고 판단합니다.
- 중복을 방지하기 위해 Python의 `set` 자료형 단위로 고유 뉴런 정보(`layer_name + 인덱스`)를 저장하여, 전체 네트워크상에서 몇 개의 은닉 뉴런 채널을 성공적으로 찔러보았는지 계산합니다.

### 3. 복합 목적 함수 계산 (Joint Loss Function)
알고리즘의 주 목표점(Loss)은 크게 두 가지 요소의 합으로 구성됩니다.
```python
# 1. 모델 예측 불일치 유도 (Difference Loss)
loss_diff = out2[0, orig_label] - out1[0, orig_label] * weight_diff

# 2. 뉴런 커버리지 극대화 유도 (Neuron Coverage Loss)
loss1_neuron = sum(act.mean() for act in model1_wrapper.activations.values())
# ... (loss2_neuron 도 동일)

# 최종 목적 함수 결합
loss = loss_diff + weight_nc * (loss1_neuron + loss2_neuron)
```
- **Difference Loss**: Model 2는 가급적 원본 정답 클래스의 확률을 유지하도록 놔두고, Model 1의 원본 정답 확률에는 마이너스를 취해 정답으로부터 멀어지게끔 의도합니다.
- **Coverage Loss**: 네트워크 전체 계층들의 평균 활성도를 더해줌으로써, 이전까지 억눌려 있던 미활성 뉴런들의 값이 커지는 방향으로 보상을 줍니다.

### 4. 경사 상승법을 통한 이미지 변형 (Gradient Ascent)
딥러닝 모델의 일반적인 학습이 "이미지를 고정하고 모델의 가중치를 업데이트(Gradient Descent)"하는 것이라면, DeepXplore 코드는 **"모델 가중치는 고정하고(평가 모드), 거꾸로 이미지 픽셀값을 업데이트"**합니다.
```python
loss.backward()
img += step_size * torch.sign(img.grad)
```
입력 이미지(`img`) 자체에 `requires_grad=True`를 설정한 뒤 `loss.backward()`를 호출하여 이미지 픽셀들에 대한 기울기를 구합니다. 이후 목표 함수(불일치 및 커버리지 증가)를 향해 나아가도록 기울기의 부호(`torch.sign`) 방향으로 이미지 픽셀에 초미세 노이즈(Step size=0.01)를 얹으며 반복(Iteration) 업데이트합니다. 이 과정을 통해 육안으로는 식별하기 어려운 노이즈가 더해지면서 최종적으로 두 모델의 예측 불일치(Discrepancy)가 터지는 이미지가 완성됩니다.
