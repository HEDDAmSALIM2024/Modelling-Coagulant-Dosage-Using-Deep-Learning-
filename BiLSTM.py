import scipy.stats as stats
from scipy.stats import zscore
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
import numpy as np
import pandas
import statistics
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import time
from sklearn.metrics  import mean_absolute_error
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Flatten
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Charger les données à partir du fichier Excel
data1 = pd.read_excel('COAGULANT DOSAGE.xlsx', sheet_name='TRAINING')
data2 = pd.read_excel('COAGULANT DOSAGE.xlsx', sheet_name='VALIDATION')



# Les 6 premières colonnes sont les caractéristiques et la dernière colonne est la cible (Q(t))
X1 = data1.iloc[:, :10].values  # Les 6 premières colonnes comme caractéristiques
Y1 = data1.iloc[:, 10].values   # Dernière colonne comme cible (Q(t))
X2 = data2.iloc[:, :10].values  # Les 6 premières colonnes comme caractéristiques
Y2 = data2.iloc[:, 10].values   # Dernière colonne comme cible (Q(t))
"""

# Les 6 premières colonnes sont les caractéristiques et la dernière colonne est la cible (Q(t))
X1 = data1.iloc[:, [0,1,2,6,8]].values  # Les 6 premières colonnes comme caractéristiques
Y1 = data1.iloc[:, 10].values   # Dernière colonne comme cible (Q(t))
X2 = data2.iloc[:, [0,1,2,6,8]].values  # Les 6 premières colonnes comme caractéristiques
Y2 = data2.iloc[:, 10].values   # Dernière colonne comme cible (Q(t))

"""
# Normaliser les données
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X1 = scaler_X.fit_transform(X1)
Y1 = scaler_Y.fit_transform(Y1.reshape(-1, 1))
X2 = scaler_X.fit_transform(X2)
Y2 = scaler_Y.fit_transform(Y2.reshape(-1, 1))

# Diviser les données manuellement en respectant l'ordre : 70 % pour entraînement et 30 % pour validation
x_train,x_test  =X1, X2
y_train,y_test  =Y1, Y2

# Restructurer les données d'entrée pour la couche LSTM (6 étapes de temps, 1 caractéristique par étape de temps)
x_train_reshaped = x_train.reshape((x_train.shape[0],  x_train.shape[1], 1))
x_test_reshaped  = x_test.reshape ((x_test.shape [0],  x_test.shape [1], 1))


# Define the BiLSTM model
model = tf.keras.Sequential()
opt = Adam(learning_rate=0.005)
model.add(Bidirectional(LSTM(300, dropout=0.10, recurrent_dropout=0.10,return_sequences=False), input_shape=(10, 1)))  # 6 time steps, 1 feature per step
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=opt, loss='mse')

# Entraîner le modèle
history = model.fit(x_train_reshaped, y_train, epochs=50 , batch_size=10, validation_data=(x_test_reshaped, y_test))
# Évaluer le modèle sur les données de validation
val_loss = model.evaluate(x_test_reshaped, y_test)
print(f'Validation Loss: {val_loss}')

tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# Prédire sur l'ensemble de validation et d'entraînement
Y_pred_val   = model.predict(x_test_reshaped)
Y_pred_train = model.predict(x_train_reshaped)

# Inverser la normalisation des prédictions et des valeurs réelles si nécessaire
Y_pred_val_original    = scaler_Y.inverse_transform(Y_pred_val)
Y_val_original         = scaler_Y.inverse_transform( y_test)
Y_pred_train_original  = scaler_Y.inverse_transform(Y_pred_train)
Y_train_original       = scaler_Y.inverse_transform(y_train)

# Calculer les métriques de performance
def print_performance_metrics(y_true, y_pred, dataset_type=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Performance metrics for {dataset_type} data:")
    print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
    print("-" * 30)

# Imprimer les métriques de performance pour les données d'entraînement et de validation
print_performance_metrics(Y_train_original, Y_pred_train_original, "Training")
print_performance_metrics(Y_val_original, Y_pred_val_original, "Validation")

# Visualiser l'historique d'entraînement
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


