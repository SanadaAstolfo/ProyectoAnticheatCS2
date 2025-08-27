import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Iniciando Proceso Completo con Modelo Híbrido ---")
try:
    df = pd.read_csv('subset_cs2cd.csv')
    print(f"Paso 1: Dataset cargado exitosamente. Forma: {df.shape}")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'dataset_con_features_prekill.csv'.")
    exit()

print("\nPaso 2: Iniciando limpieza y preprocesamiento de datos...")
if 'aim_punch_angle' in df.columns:
    df['aim_punch_angle'] = df['aim_punch_angle'].astype(str).fillna('[0 0]')
    split_df = df['aim_punch_angle'].str.strip('[]').str.split(expand=True)
    df[['aim_punch_x', 'aim_punch_y']] = split_df.iloc[:, :2]
    df['aim_punch_x'] = pd.to_numeric(df['aim_punch_x'], errors='coerce').fillna(0)
    df['aim_punch_y'] = pd.to_numeric(df['aim_punch_y'], errors='coerce').fillna(0)
    df = df.drop(columns=['aim_punch_angle'])

for col in ['is_walking', 'FIRE', 'is_scoped']:
    if col in df.columns:
        df[col] = (df[col].notna() & (df[col] != 'False')).astype(int)

columnas_a_eliminar = ['steamid', 'spotted', 'approximate_spotted_by', 'map', 'server', 'avg_rank', 'match_making_type']
df = df.drop(columns=columnas_a_eliminar, errors='ignore')

columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
for col in columnas_numericas:
    if df[col].isnull().sum() > 0:
        mediana = df[col].median()
        df[col].fillna(mediana, inplace=True)
print("Limpieza y preprocesamiento finalizados.")

print("\nPaso 3: Escalando características...")
features = [col for col in df.columns if col != 'is_cheater']
X = df[features]
y = df['is_cheater']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X).astype('float32')
print("Características escaladas a float32.")

print("\nPaso 4: Dividiendo datos y preparando generador...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

TIME_STEPS = 32
BATCH_SIZE = 128

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_data, y_data, time_steps, batch_size):
        self.X, self.y = X_data, y_data
        self.time_steps = time_steps
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.X) - self.time_steps) / self.batch_size)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        
        Xs, ys = [], []
        for i in range(start, end):
            if i >= self.time_steps:
                Xs.append(self.X[i-self.time_steps:i])
                ys.append(self.y[i])
        return np.array(Xs), np.array(ys)

train_generator = DataGenerator(X_train, y_train.to_numpy(), TIME_STEPS, BATCH_SIZE)
print(f"Generador de datos de entrenamiento creado con {len(train_generator)} lotes.")

def create_test_sequences(X, y, time_steps=32):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

try:
    X_test_seq, y_test_seq = create_test_sequences(X_test, y_test, TIME_STEPS)
    print(f"Conjunto de prueba creado en memoria. Forma: {X_test_seq.shape}")
    VALIDATION_DATA = (X_test_seq, y_test_seq)
except MemoryError:
    print("Advertencia: El conjunto de prueba es demasiado grande. Se usará un generador para validación.")
    VALIDATION_DATA = DataGenerator(X_test, y_test.to_numpy(), TIME_STEPS, BATCH_SIZE)

print("\nPaso 5: Construyendo la arquitectura del modelo híbrido...")
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(TIME_STEPS, X_train.shape[1])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(GRU(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("\nPaso 6: Entrenando el modelo...")
history = model.fit(train_generator, epochs=5, validation_data=VALIDATION_DATA, verbose=1)
print("¡Entrenamiento completado!")

print("\nPaso 7: Evaluando el rendimiento del modelo...")
y_pred_prob = model.predict(X_test_seq)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n--- Reporte de Clasificación (Híbrido CNN+GRU) ---")
print(classification_report(y_test_seq, y_pred))

cm = confusion_matrix(y_test_seq, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Legítimo', 'Tramposo'], yticklabels=['Legítimo', 'Tramposo'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión (Híbrido CNN+GRU)')
plt.show()

print("\n--- Proceso Completo Finalizado ---")