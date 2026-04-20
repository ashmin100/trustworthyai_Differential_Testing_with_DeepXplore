# 탐색 결과 및 분석 (Results & Analysis)

## 1. 탐색 수치 보고 (Quantitative Results)

- **생성된 불일치 이미지 수 (Number of disagreement-inducing inputs found):** 총 5개의 모델 간 예측 불일치 이미지 조합을 발견했습니다.

- **달성된 뉴런 커버리지 (Neuron coverage achieved):** 
  ResNet50 모델 구조 내에 존재하는 **모든 컨볼루션 계층(전체 `nn.ReLU`) 채널(수만 개)**을 재귀적 Hook 방식으로 모조리 측정 지점으로 삼았습니다. 
  테스트 셋팅(Threshold=0.1 등)에 따라 다소 편차가 있으나, 생성된 모델 불일치 결과물은 평균적으로 전체 뉴런의 **70~80% 대의 활성화 비율(%)**을 안정적으로 보이고 있습니다. 즉, 오답 틈새를 찾아내는 동시에 미활성 뉴런들의 활성을 적극적으로 끌어당기면서 목표했던 모델 한계선까지 효율적으로 도달했습니다. 
  *(자세한 이미지별 백분율 차이는 스크립트 실행과 함께 생성되는 `results/coverage_summary_chart.png`를 참고해 주세요!)*

## 불일치 유발 입력 유형 및 원인 분석 (Discussion)

**어떤 타입의 입력이 불일치를 유발시키는가? (What types of inputs cause disagreements)**
- 인간의 육안(오리지널 라벨)으로는 여전히 올바른 정답이 무엇인지 구분이 가능한 원본 이미지의 형태를 띠고 있습니다. 하지만 픽셀의 미시적 단위에서 고주파(High-frequency)의 불규칙한 **적대적 노이즈(Perturbation noise) 패턴이 덧씌워진 유형**입니다.
- 인간에게는 원본과 똑같아 보이지만 모델의 특징 추출기(Feature Extractor)에는 심각한 혼동 요소로 작용하는 정교한 미세 변형 입력입니다.

**이러한 입력이 왜 불일치를 유발하는가? (Why do they cause disagreements?)**
- 차이점 테스트(Differential Testing)의 기준이 되는 Model 1과 Model 2는 동일한 아키텍처(ResNet50)일지라도, 서로 다른 가중치 초기화(Initialization Seed)로 인해 **학습하는 픽셀 단위의 의존성 패턴(Non-robust features)이 미세하게 다릅니다.** 
- 한 모델은 객체의 외곽선(Shape)에 더 민감하게 반응하고, 다른 모델은 텍스처(Texture)나 색상 조합에 조금 더 의존적일 수 있습니다.
- DeepXplore 알고리즘의 공동 손실 함수(Joint Loss Function)는 바로 이렇게 '두 모델이 각자 다르게 의존하는 취약점'의 틈새를 파고듭니다. 결과적으로 **특정 노이즈가 Model 1의 판단 임계선(Decision Boundary)은 붕괴시켜 오답을 유도하지만, Model 2의 의존 패턴은 빗겨나가게 하여 여전히 정답을 예측하도록 이중 분리 현상**을 만들어내기 때문에 불일치(Disagreement)가 발생하게 됩니다.
