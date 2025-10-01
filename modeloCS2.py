import pandas as pd
import numpy as np
import time
import os
import gc
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("--- Iniciando Proceso ---", flush=True)
try:
    df = pd.read_csv('subset_cs2cd.csv', low_memory=False)
    print(f"Paso 1: Dataset cargado. Forma: {df.shape}", flush=True)
except FileNotFoundError:
    print("Error: No se encontró el archivo del dataset.", flush=True)
    exit()

print("\nPaso 2: Limpieza y preprocesamiento...", flush=True)
start_time = time.time()

df['new_kill'] = df.groupby('steamid')['kills_total'].diff()
kill_events_df = df[df['new_kill'] == 1].copy()
df['variance_pre_kill'] = 0.0
df['suma_abs_pre_kill'] = 0.0
for index, kill_event in kill_events_df.iterrows():
    steam_id, kill_tick = kill_event['steamid'], kill_event['tick']
    start_tick = kill_tick - 192
    pre_kill_window_df = df[(df['steamid'] == steam_id) & (df['tick'] >= start_tick) & (df['tick'] < kill_tick)]
    if not pre_kill_window_df.empty:
        variance = np.nansum([pre_kill_window_df['usercmd_mouse_dx'].var(), pre_kill_window_df['usercmd_mouse_dy'].var()])
        abs_sum = pre_kill_window_df['usercmd_mouse_dx'].abs().sum() + pre_kill_window_df['usercmd_mouse_dy'].abs().sum()
        df.loc[index, 'variance_pre_kill'] = variance
        df.loc[index, 'suma_abs_pre_kill'] = abs_sum
df = df.drop(columns=['new_kill'])

if 'aim_punch_angle' in df.columns:
    df['aim_punch_angle'] = df['aim_punch_angle'].astype(str).fillna('[0 0]')
    split_df = df['aim_punch_angle'].str.strip('[]').str.split(expand=True)
    df[['aim_punch_x', 'aim_punch_y']] = split_df.iloc[:, :2]
    df[['aim_punch_x', 'aim_punch_y']] = df[['aim_punch_x', 'aim_punch_y']].apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.drop(columns=['aim_punch_angle'])

for col in ['is_walking', 'FIRE', 'is_scoped']:
    if col in df.columns:
        df[col] = (df[col].notna() & (df[col] != 'False')).astype(int)

columnas_para_encoder = ['map', 'server', 'avg_rank', 'match_making_type']
df = pd.get_dummies(df, columns=columnas_para_encoder, drop_first=True, dtype=float)

df_cleaned = df.drop(columns=['spotted', 'approximate_spotted_by'], errors='ignore')

cols = df_cleaned.columns
if cols.duplicated().any():
    df_cleaned = df_cleaned.loc[:, ~cols.duplicated()]

columnas_numericas = df_cleaned.select_dtypes(include=np.number).columns
for col in columnas_numericas:
    if df_cleaned[col].isnull().any():
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
print(f"Limpieza finalizada en {time.time() - start_time:.2f} segundos.", flush=True)

print("\nPaso 3: Dividiendo datos por jugador...", flush=True)
features = [col for col in df_cleaned.columns if col not in ['is_cheater', 'steamid']]
X = df_cleaned[features]
y = df_cleaned['is_cheater']
groups = df_cleaned['steamid']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nPaso 4: Escalando características...", flush=True)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')

print("\nPaso 5: Preparando generadores de datos...", flush=True)
TIME_STEPS = 32
BATCH_SIZE = 256

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_data, y_data, time_steps, batch_size):
        self.X, self.y, self.time_steps, self.batch_size = X_data, y_data, time_steps, batch_size
    def __len__(self):
        return int(np.floor((len(self.X) - self.time_steps) / self.batch_size))
    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        Xs, ys = [], []
        for i in range(start, end):
            if i >= self.time_steps:
                Xs.append(self.X[i - self.time_steps:i])
                ys.append(self.y[i])
        return np.array(Xs), np.array(ys)

train_generator = DataGenerator(X_train_scaled, y_train.to_numpy(), TIME_STEPS, BATCH_SIZE)
validation_generator = DataGenerator(X_test_scaled, y_test.to_numpy(), TIME_STEPS, BATCH_SIZE)
print("Generadores creados.", flush=True)

print("Liberando memoria...", flush=True)
del df, df_cleaned, X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
gc.collect()

print("\nPaso 6: Construyendo arquitectura...", flush=True)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))
print(f"Pesos de clase calculados: {class_weights}", flush=True)

model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(TIME_STEPS, train_generator.X.shape[1])),
    MaxPooling1D(pool_size=2), Dropout(0.3), GRU(units=50), Dropout(0.3),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("\nPaso 7: Entrenando el modelo...", flush=True)
history = model.fit(train_generator, epochs=10, validation_data=validation_generator, class_weight=class_weights, verbose=1)
print("¡Entrenamiento completado!", flush=True)
model.save('anticheat_model_final.keras')
print("Modelo guardado.", flush=True)

print("\nPaso 8: Evaluando el rendimiento...", flush=True)
y_pred_prob = model.predict(validation_generator)

y_true = []
for i in range(len(validation_generator)):
    _, y_batch = validation_generator[i]
    y_true.extend(y_batch)
y_true = np.array(y_true)
y_pred = (y_pred_prob[:len(y_true)] > 0.5).astype(int)

print("\n--- Reporte de Clasificación ---", flush=True)
print(classification_report(y_true, y_pred, target_names=['Legítimo (0)', 'Tramposo (1)']))

cm = confusion_matrix(y_true, y_pred)
print("\n--- Matriz de Confusión ---", flush=True)
print(f"Real Legítimo: {cm[0][0]:>7d} | {cm[0][1]:>8d}", flush=True)
print(f"Real Tramposo: {cm[1][0]:>7d} | {cm[1][1]:>8d}", flush=True)

print("\n--- Análisis con Curva ROC ---", flush=True)
fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:len(y_true)])
auc_score = roc_auc_score(y_true, y_pred_prob[:len(y_true)])
print(f"Área Bajo la Curva (AUC): {auc_score:.4f}", flush=True)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC del Modelo Final')
plt.legend()
plt.grid()
plt.savefig('curva_roc_final.png')
plt.close()
print("Curva ROC guardada.", flush=True)

print("\n--- Proceso Completo Finalizado ---", flush=True)