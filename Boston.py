from keras.datasets import boston_housing
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_testtargets) = (boston_housing.load_data())  # 독립변수(x), 피쳐
# 독립변수에 의해 결정되는 종속 변수(y값)
# print(train_data)
# target는 천달러 단위
print(train_data[0, :])
# 특성 중앙을 0으로 표준편차가1로 만들어줌


mean = train_data.mean(axis=0)  # 컬럼별 평균구하기  .mean은 평균구하는것
train_data -= mean  # 컬럼별 평군을 빼기
std = train_data.std(axis=0)  # 컬럼별 표준편차 구하기  .std는 표준편차 구하는것
train_data /= std  # 컬럼별 표준편차로 나눠주기

test_data -= mean  # 테스트 데이터에적용
test_data /= std  # 테스트 데이터에적용


# 모델 아키텍처 (출력 형태)
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop",
                  loss="mse",
                  metrics=["mae"])
    return model


# K-Fold Cross Validation 적용해서 훈련검증을 함(데이터 수가 적음)
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
all_scores = []
for i in range(k):
    print(f"#{i}번째 폴드 처리중")
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_target = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()  # 케라스 모델 구성
    history = model.fit(partial_train_data, partial_train_target,  # 모델훈련(verbose=0이므로 훈련과정이 출력되지 않습니다.)
                        epochs=num_epochs,
                        validation_data=(val_data, val_targets),  # 검증세트로 모델평가
                        batch_size=16,
                        verbose=1)

mae_history = history.history['val_mae']
all_mae_histories.append(mae_history)
avgerage_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
truncated_mae_history = avgerage_mae_history[10:]
plt.figure(figsize=(16, 10))
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("validation MAE")
plt.savefig("fig.png")