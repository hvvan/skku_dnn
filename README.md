# skku_dnn
심층신경망개론 5차 과제
2018313414 김지환
 사용한 팀 이름: 캔병플라스틱 
이번 5차 과제 보고서의 목차는 아래와 같습니다.
1.	모델 예측 성능 향상 방안
2.	아쉬운 점
  우선 모델 예측 성능을 향상시킨 방안에 대해 서술하고, 구현 상에서 아쉬웠던 점을 이어 말씀드리도록 하겠습니다.
1.	모델 예측 성능 향상 방안
제가 Baseline으로부터 모델 예측 성능을 향상시킨 방법은 아래와 같습니다.
-	Bidirectional = True
-	Learning rate scheduler -> CosineAnnealingWarmRestarts
-	Increase batch size (32 -> 64)
-	Increase teacher forcing ratio -> teacher forcing ratio decay
-	Reg_lamda = 1e-2
-	Epoch = 160
a.	Bidirectional = True
우선 lstm의 bidirectional option을 true로 설정하였습니다. 이렇게 설정한 이유는 정방향의 정보와 역방향의 정보를 모두 갖게 하여 long term dependency 문제를 해소하기 위함입니다.
b.	Learning rate scheduler -> CosineAnnealingWarmRestarts
그리고 learning rate scheduler를 도입하여 loss가 충분히 낮아진 상황에서 학습이 진행될 수 있도록 하는 learning rate scheduler를 도입하였습니다. 여러 learning rate scheduler를 사용해보았지만, 가장 성능이 좋은 것은 CosineAnnealingWarmRestarts scheduler였습니다. CosineAnnealingWarmRestarts는 일정한 주기를 가지며 설정해둔 learning rate을 차근차근 줄이는 과정을 거치도록 하는 scheduler입니다. 이 스케줄러가 좋은 이유는 중간중간 작아졌던 learning rate을 크게 키우는 방식을 사용하여 고질적인 local optima에서 탈출하여 global minima로 수렴해갈 수 있기 때문입니다. 또한 경험적인 부분이긴 하지만, adamW optimizer를 사용하면서 성능이 가장 좋았던 스케줄러가 CosineAnnealingWarmRestarts였습니다. 
정한 learning rate scheduler를 사용하여 초기 learning rate, 그리고 주기에서 가장 마지막 learning rate, 그리고 그 주기를 설정하여야 했습니다. 이때 생각한 방법으로 기존 Baseline에서 주어진 learning rate인 1e-5가 기하평균이 될 수 있도록 초기 learning rate과 마지막 learning rate을 설정해봤습니다. 학습이 진행될수록 낮은 learning rate을 가지도록 하는 것이 바람직 할 것 같아서 최솟값을 1e-5보다 10배, 100배 작아지는 상황에서 test를 진행해보았는데 1e-5보다 10배 작아지는 상황에서의 성능이 더 높았습니다. 또한 주기는 30씩 총 epoch는 150이 되도록 하였습니다. 따라서 learning rate의 초기값을 1e-4로, 그리고 1 주기에서의 마지막 learning rate의 값은 (1e-6) 이 되도록 설정하였습니다.
c.	Increase batch size (32 -> 64)
수렴이 빨리 되게 하기 위해서 batch size를 키웠습니다. Batch size를 더 키웠을 때는 어떤 값이 나오는지 궁금하여 실험을 해보고자 하였으나, GPU의 용량 issue로 확인할 수 없었습니다.
d.	Increase teacher forcing ratio -> teacher forcing ratio decay
Teacher forcing ratio를 epoch가 진행될수록 줄여주었습니다. 이 기법을 사용하여 학습이 진행되면서 정답 데이터에 대한 의존성을 줄여주기 위함입니다. Learning rate scheduler와 비슷한 개념으로, 학습이 진행되면서 overfitting이 되지 않고 강건하게 학습이 진행될 수 있는 방법을 생각해보다가, teacher forcing ratio를 학습이 진행되면서 줄여주는 방법을 생각해보았습니다. Teacher forcing ratio 초기값을 각각 0.7, 0.5, 0.3에서 시작하여 0.1까지 도달하도록 설정하였을 때,  가장 높은 성능을 보이는 것은 0.3에서 시작하는 것이었습니다.
e.	Reg_lambda: 1e-2
Reg_lambda값을 1e-2로 둠으로써 이전 학습 방향의 값을 유지할 수 있도록 하였습니다. 이 방법을 사용하였을 때의 이점은 보다 강건하게 모델을 학습시킬 수 있다는 점입니다. 실제로 학습을 진행하며 loss가 튕기는 것이 없이 학습이 되는 것을 확인할 수 있었습니다. Optimizer를 AdamW를 활용하여 사용하였기에 유효한 hyperparameter였다고 생각합니다.

이렇게 위에서 말씀드린 5개의 상황을 모두 종합하면 제가 구상한 코드가 나오게 됩니다.
이어서 아쉬웠던 점을 작성하도록 하겠습니다.
2.	아쉬운 점
Attention을 사용하여 어느 부분을 중요하게 볼 것인지를 확인하는 코드를 구현하고 싶었으나, 생각보다 코드 구현이 복잡하게 되어있어 마무리 짓지 못한 것이 아쉬웠습니다. 그러나 이에 대한 아쉬운 점을 teacher forcing ratio decay 기법을 통해 어느정도 상쇄할 수 있었던 것 같아 만족스럽습니다. 컴퓨팅 리소스가 부족하여 코드를 많이 돌려보지 못한 점은 아쉽지만, 좋은 결과가 나올 수 있도록 치열하게 고민할 수 있는 시간을 가진 것 같아 개인적으로 만족스러웠던 시간이었습니다. 이상으로 보고서를 마무리하도록 하겠습니다. 감사합니다.
