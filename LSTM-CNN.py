
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
from sklearn.metrics  import mean_absolute_error
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
X_test = data_val.iloc[:, [0,1,2,7,8]].values
Y_test = data_val.iloc[:, 10].values
"""
# Normalisation des données
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1))
X_test = scaler_X.transform(X_test)
Y_test = scaler_Y.transform(Y_test.reshape(-1, 1))

# Reshaping pour Conv1D et LSTM (n_samples, n_timesteps, n_features)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped  = X_test.reshape ((X_test.shape[0],  X_test.shape [1], 1))

# Définir le modèle CNN-LSTM
model = Sequential()

# Partie CNN
model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
# Partie LSTM
model.add(LSTM(300, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
# Couche Dense
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
# Compiler le modèle
model.compile(optimizer='adam', loss='mse')

# Entraîner le modèle
history = model.fit(X_train_reshaped, Y_train, epochs=200, batch_size=10, validation_data=(X_test_reshaped, Y_test))

# Évaluer le modèle sur les données de validation
val_loss = model.evaluate(X_test_reshaped, Y_test)
print(f'Validation Loss: {val_loss}')

tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# Prédire sur l'ensemble de validation et d'entraînement
Y_pred_val   = model.predict(X_test_reshaped)
Y_pred_train = model.predict(X_train_reshaped)

# Inverser la normalisation des prédictions et des valeurs réelles si nécessaire
Y_pred_val_original = scaler_Y.inverse_transform(Y_pred_val)
Y_val_original = scaler_Y.inverse_transform( Y_test)
Y_pred_train_original = scaler_Y.inverse_transform(Y_pred_train)
Y_train_original = scaler_Y.inverse_transform(Y_train)

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








