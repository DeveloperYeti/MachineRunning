from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
X = pd.DataFrame(data.data)
Y = pd.DataFrame(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

ScalerX = StandardScaler()
X_train = ScalerX.fit_transform(X_train)
X_test = ScalerX.transform(X_test)
print()
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))

inputs = keras.Input(shape=(30,1,1))
x = layers.Conv2D(filters=32, kernel_size=(3,1) , activation="relu")(inputs)
x = layers.Conv2D(filters=64, kernel_size=(3,1) , activation="relu")(x)
x = layers.Conv2D(filters=32, kernel_size=(3,1) , activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs , outputs = outputs)

model.compile(optimiezer = "rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(X_test , y_test)
print(f"테스트 정확도 {test_acc:.3f}")


