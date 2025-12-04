import pandas as pd
import numpy as np
import time
import os
import gc
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("--- Iniciando Proceso ---", flush=True)
print("ADVERTENCIA: Este script intenta cargar todo el dataset en memoria.", flush=True)
print("Si tienes problemas de memoria, usa 'modeloCS2umbral.py' en su lugar.", flush=True)

print("\nPaso 1: Cargando dataset completo...", flush=True)
try:
    df = pd.read_csv('subset_cs2cd.csv', low_memory=False)
    print(f"Dataset completo cargado. Forma: {df.shape}", flush=True)
except FileNotFoundError:
    print("Error: No se encontró el archivo dataset.", flush=True)
    exit()
except MemoryError:
    print("\nERROR: Memoria RAM insuficiente para cargar dataset completo.", flush=True)
    print("   Solución: Ejecuta 'modeloCS2umbral.py' que usa solo 3M filas.", flush=True)
    exit()
except Exception as e:
    print(f"Error al cargar dataset: {e}", flush=True)
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

columnas_categoricas_eliminar = ['map', 'server', 'avg_rank', 'match_making_type']
df = df.drop(columns=columnas_categoricas_eliminar, errors='ignore')

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
joblib.dump(features, 'columnas_numericas.pkl')
print(f"✅ Columnas del modelo guardadas: {len(features)} features", flush=True)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("\nPaso 4: Identificando eventos de kill...", flush=True)
kill_ilocs_train = np.where(X_train['variance_pre_kill'] > 0)[0]
kill_ilocs_test = np.where(X_test['variance_pre_kill'] > 0)[0]
print(f"Kills encontrados en set de entrenamiento: {len(kill_ilocs_train)}", flush=True)
print(f"Kills encontrados en set de prueba: {len(kill_ilocs_test)}", flush=True)

print("\nPaso 5: Escalando características...", flush=True)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')
print("Scaler guardado como 'scaler_entrenado.pkl'", flush=True)

print("\nPaso 6: Creando ventanas de contexto (clips) alrededor de los kills...", flush=True)
WINDOW_BEFORE = 224
WINDOW_AFTER = 32
WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER
BATCH_SIZE = 256

def create_kill_windows(X_scaled_data, y_data, kill_ilocs):
    X_windows, y_windows = [], []
    y_data_np = y_data.to_numpy()
    
    for i in kill_ilocs:
        if i < WINDOW_BEFORE or (i + WINDOW_AFTER) > len(X_scaled_data):
            continue
            
        window = X_scaled_data[i - WINDOW_BEFORE : i + WINDOW_AFTER]
        X_windows.append(window)
        y_windows.append(y_data_np[i])
        
    return np.array(X_windows), np.array(y_windows)

X_train_windows, y_train_windows = create_kill_windows(X_train_scaled, y_train, kill_ilocs_train)
X_test_windows, y_test_windows = create_kill_windows(X_test_scaled, y_test, kill_ilocs_test)

print(f"Dataset de entrenamiento creado: {X_train_windows.shape}", flush=True)
print(f"Dataset de prueba creado: {X_test_windows.shape}", flush=True)

print("Liberando memoria de dataframes originales...", flush=True)
del df, df_cleaned, X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, kill_ilocs_train, kill_ilocs_test
gc.collect()

print("\nPaso 8: Construyendo arquitectura del modelo...", flush=True)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_windows), y=y_train_windows)
class_weights = dict(enumerate(weights))
print(f"Pesos de clase (para ventanas) calculados: {class_weights}", flush=True)

num_features = X_train_windows.shape[2]

model = Sequential([
    Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(WINDOW_SIZE, num_features)),
    MaxPooling1D(pool_size=2),
    GRU(units=100, return_sequences=True),
    Dropout(0.5),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),
    GRU(units=50),
    Dropout(0.3),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("\nPaso 9: Entrenando el modelo (sobre ventanas de contexto)...", flush=True)
history = model.fit(
    X_train_windows, 
    y_train_windows,
    epochs=10, 
    validation_data=(X_test_windows, y_test_windows),
    class_weight=class_weights,
    batch_size=BATCH_SIZE,
    verbose=1
)
print("¡Entrenamiento completado!", flush=True)
model.save('anticheat_model_kill_focused.keras')
print("Modelo guardado.", flush=True)

print("\nPaso 10: Evaluando el rendimiento...", flush=True)
y_pred_prob = model.predict(X_test_windows)
y_true = y_test_windows

print("\n--- Reporte de Clasificación (Umbral 0.8) ---", flush=True)
y_pred_default = (y_pred_prob > 0.8).astype(int)
print(classification_report(y_true, y_pred_default, target_names=['Legítimo (0)', 'Tramposo (1)']))

cm_default = confusion_matrix(y_true, y_pred_default)
print("\n--- Matriz de Confusión (Umbral 0.8) ---", flush=True)
print(f"Real Legítimo: {cm_default[0][0]:>7d} | {cm_default[0][1]:>8d}", flush=True)
print(f"Real Tramposo: {cm_default[1][0]:>7d} | {cm_default[1][1]:>8d}", flush=True)

print("\n--- Análisis con Curvas de Evaluación ---", flush=True)
fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_prob)
auc_score = roc_auc_score(y_true, y_pred_prob)
print(f"Área Bajo la Curva ROC (AUC): {auc_score:.4f}", flush=True)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'r--', label='Clasificador Aleatorio')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
plt.title('Curva ROC (Enfoque Kills)')
plt.legend()
plt.grid()
plt.savefig('curva_roc_kills.png')
plt.close()
print("Curva ROC guardada como 'curva_roc_kills.png'.", flush=True)

precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
avg_precision = average_precision_score(y_true, y_pred_prob)
print(f"Precisión Promedio (AP): {avg_precision:.4f}", flush=True)

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, label=f'Curva P-R (AP = {avg_precision:.2f})')
plt.xlabel('Recall (Sensibilidad)')
plt.ylabel('Precisión')
plt.title('Curva Precisión-Recall (Enfoque Kills)')
plt.legend()
plt.grid()
plt.savefig('curva_precision_recall_kills.png')
plt.close()
print("Curva Precisión-Recall guardada como 'curva_precision_recall_kills.png'.", flush=True)

print("\n--- Paso 11: Calibración Automática del Umbral ---", flush=True)
f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
best_f1_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[best_f1_idx]
print(f"Umbral óptimo que maximiza el F1-Score: {optimal_threshold:.4f}", flush=True)

y_pred_optimizado = (y_pred_prob > optimal_threshold).astype(int)
print("\n--- Reporte de Clasificación (con Umbral Optimizado) ---", flush=True)
print(classification_report(y_true, y_pred_optimizado, target_names=['Legítimo (0)', 'Tramposo (1)']))

cm_optimizado = confusion_matrix(y_true, y_pred_optimizado)
print("\n--- Matriz de Confusión (con Umbral Optimizado) ---", flush=True)
print(f"Real Legítimo: {cm_optimizado[0][0]:>7d} | {cm_optimizado[0][1]:>8d}", flush=True)
print(f"Real Tramposo: {cm_optimizado[1][0]:>7d} | {cm_optimizado[1][1]:>8d}", flush=True)

print("\n--- Proceso Completo Finalizado ---", flush=True)