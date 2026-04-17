# 과제 2 - 문제 1: DeepXplore를 활용한 차이점 테스트(Differential Testing)

이 저장소는 DeepXplore의 개념을 차용하여 차이점 테스트를 수행하는 과제 2의 문제 1번 해답 코드를 포함하고 있습니다.

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
과제 명세에 연결되어 있는 원래 DeepXplore 코드는 더 이상 잘 쓰이지 않는 오래된 Keras와 TensorFlow 1.x 프레임워크를 기반으로 작성되어 있었습니다. "최신 프레임워크에 맞게 수정하라(adapt it to work with current frameworks)"는 과제의 요구 사항을 충족하기 위하여, DeepXplore의 핵심 로직 전체를 **PyTorch** 딥러닝 프레임워크에 맞게 전부 재작성(Porting)했습니다. 

- **PyTorch 적용(Adaptation)**: 기존의 `gen_diff.py` 스크립트를 `test.py`로 대체했습니다. `autograd` 및 `register_forward_hook` 함수를 사용하여 중간 레이어 활성화 값을 가로채어 뉴런 커버리지(Neuron Coverage)를 계산하고, 미분을 이용한 경사 상승법(Gradient Ascent)으로 이미지를 변형하는 로직을 PyTorch에 맞게 구현했습니다.
- **CIFAR-10용 ResNet50 모델 구성**: `torchvision.models.resnet50`의 구조(처음 Convolution 계층과 마지막 fully-connected 계층)를 32x32 크기의 CIFAR-10 이미지 및 10개 클래스 출력에 맞추어 변형했습니다. 두 모델이 서로 의미 있는 차이를 보이도록 무작위 시드(Seed)를 각각 да르게 주어 초기화한 뒤(각각 `model1`과 `model2`), 자동으로 1에포크(Epoch) 동안 짧게 학습을 진행하여 최소한의 분류 능력을 갖추도록 만듭니다. (미리 저장된 `model1.pt`, `model2.pt`가 없을 경우 실행 시 자동으로 이를 새로 구성하고 훈련합니다).
- **뉴런 커버리지(Neuron Coverage)**: 입력 값이 두 모델의 깊숙한 컨볼루션 계층(예: `layer4_relu` 출력)을 어떻게 활성화하는지 감시하여 일정 임계값을 상회하는 뉴런 수를 동적으로 계산합니다.

## 실행 결과
`test.py`를 실행하면 `results/` 폴더 내에 두 모델 간 예측 결과 불일치를 유도한 이미지 결과물이 5개 이상 생성됩니다.
각 이미지는 아래 3가지 요소를 시각적으로 보여줍니다:
1. **원본 이미지 (Original)**
2. **모델 간 예측 차이를 발생시킨 변형 이미지 (Perturbed)**
3. **더해진 노이즈 변화량 (Perturbation Noise)**
