from sklearn.metrics import r2_score
import numpy as np
import pandas
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import keras
import numpy as np
from keras import layers
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv1D, Flatten, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras import backend as K


# Charger les données depuis un fichier Excel
data = pd.read_excel     ('COAGULANT DOSAGE.xlsx', sheet_name='TRAINING')  
data_val = pd.read_excel ('COAGULANT DOSAGE.xlsx', sheet_name='VALIDATION')



# Préparation des données
# Supposons que les 10 premières colonnes sont des caractéristiques et la dernière colonne est la cible
X_train = data.iloc[:, :10].values  # Caractéristiques
Y_train = data.iloc[:, 10].values   
X_test  = data_val.iloc[:, :10].values
Y_test  = data_val.iloc[:, 10].values

"""
# Exemple: Sélectionner uniquement la 2ème (index 1) et la 5ème (index 4) colonnes comme entrées
X_train = data.iloc[:, [0,1,2,6,8]].values  # Caractéristiques (2ème et 5ème colonnes)
Y_train = data.iloc[:, 10].values      # Cible 
X_test = data_val.iloc[:, [0,1,2,6,8]].values
Y_test = data_val.iloc[:, 10].values
"""

# Normalisation des données
scaler_X = MinMaxScaler(feature_range=(0 , 1))
scaler_Y = MinMaxScaler(feature_range=(0 , 1))

X_train = scaler_X.fit_transform(X_train)
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1))
X_test = scaler_X.transform(X_test)
Y_test = scaler_Y.transform(Y_test.reshape(-1, 1))

# Reshaping pour CNN (n_samples, n_timesteps, n_features)
X_train_reshaped = X_train.reshape((X_train.shape[0],  X_train.shape[1],1))  # (samples, timesteps, features)
X_test_reshaped  = X_test.reshape ((X_test.shape[0],   X_test.shape [1],1))


# Définir le modèle CNN
model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.summary()

history = model.fit(X_train_reshaped, Y_train, epochs=40, batch_size=10, validation_data=(X_test_reshaped, Y_test))

# Évaluer le modèle sur les données de validation

val_loss = model.evaluate(X_test_reshaped, Y_test)
print(f"Validation Loss: {val_loss}")

# Prédire sur l'ensemble de validation
Y_pred1 = model.predict(X_train_reshaped)
Y_pred2 = model.predict(X_test_reshaped)

# Inverser la normalisation pour obtenir les valeurs d'origine
Y_pred_original1 = scaler_Y.inverse_transform(Y_pred1)
Y_pred_original2 = scaler_Y.inverse_transform(Y_pred2)
Y_train_original = scaler_Y.inverse_transform (Y_train)
Y_test_original  = scaler_Y.inverse_transform (Y_test)


# Calculer les métriques de performance
mse_train = mean_squared_error (Y_train_original, Y_pred_original1)
r2_train  = r2_score           (Y_train_original, Y_pred_original1)
print(f'MSE: {mse_train}, R^2: {r2_train}')

mse_train = mean_squared_error (Y_test_original, Y_pred_original2)
r2_train  = r2_score           (Y_test_original, Y_pred_original2)
print(f'MSE: {mse_train}, R^2: {r2_train}')

# Visualisation des résultats
plt.plot(Y_test_original,  label='True')
plt.plot(Y_pred_original2, label='Predicted')
plt.title('True vs Predicted values')
plt.legend()
plt.show()

# Visualiser l'historique de l'entraînement
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


