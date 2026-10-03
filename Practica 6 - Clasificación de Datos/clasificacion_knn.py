import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os

# Configuración estética global
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

# Carga el dataset limpio, filtra las observaciones con condición conocida
# (excluyendo 'unknown') y estructura la variable objetivo en 3 clases ordinales:
# - 'Excelente' (new, like new, excellent)
# - 'Buena' (good)
# - 'Regular / Proyecto' (fair, salvage)
def cargar_y_preparar_datos(ruta_csv: str) -> tuple:
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}")
    
    df = pd.read_csv(ruta_csv)
    
    # Excluir registros sin etiqueta de condición observable
    df_known = df[df['condition'] != 'unknown'].copy()
    
    # Extracción numérica de cilindrada ('8 cylinders' -> 8.0)
    df_known['cylinders_num'] = df_known['cylinders'].str.extract(r'(\d+)').astype(float).fillna(8)
    
    # Mapeo a 3 clases
    map_condition = {
        'new': 'Excelente',
        'like new': 'Excelente',
        'excellent': 'Excelente',
        'good': 'Buena',
        'fair': 'Regular / Proyecto',
        'salvage': 'Regular / Proyecto'
    }
    df_known['target_condition'] = df_known['condition'].map(map_condition)
    
    features = ['price', 'odometer', 'year', 'cylinders_num']
    X = df_known[features]
    y = df_known['target_condition']
    
    return df_known, X, y

# Crea la subcarpeta de gráficas si no existe
def crear_directorio_graficas(carpeta_salida: str) -> None:
    os.makedirs(carpeta_salida, exist_ok=True)

# ==============================================================================
# 1. PARTICIÓN ESTRATIFICADA Y ESCALAMIENTO DE FEATURES
# ==============================================================================

# Divide los datos en conjuntos de entrenamiento (80%) y prueba (20%) estratificados.
# Estandariza las características numéricas mediante StandardScaler (media = 0, std = 1).
def dividir_y_escalar_datos(X: pd.DataFrame, y: pd.Series, df_known: pd.DataFrame, 
                            test_size: float = 0.20, random_state: int = 42) -> tuple:
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_known.index, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Se aplica la transformación sobre test usando los parámetros aprendidos en train (evita data leakage)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, idx_train, idx_test, scaler

# ==============================================================================
# 2. BÚSQUEDA DEL K ÓPTIMO Y CURVA DE ACCURACY
# ==============================================================================

# Calcula la precisión en entrenamiento y prueba para cada hiperparámetro K.

def evaluar_curva_k(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, 
                    y_train: pd.Series, y_test: pd.Series, 
                    k_max: int = 25, carpeta_salida: str = "") -> tuple:
    k_range = list(range(1, k_max + 1))
    train_accuracies = []
    test_accuracies = []
    
    for k in k_range:
        # Ponderación por el inverso de la distancia euclidiana: w_i = 1 / d(p, q_i)
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train_scaled, y_train)
        train_accuracies.append(knn.score(X_train_scaled, y_train))
        test_accuracies.append(knn.score(X_test_scaled, y_test))
        
    best_idx = int(np.argmax(test_accuracies))
    best_k = k_range[best_idx]
    best_acc = test_accuracies[best_idx]
    
    # Graficar Curva de K vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_accuracies, marker='o', linestyle='--', color='darkblue', alpha=0.7, label='Accuracy en Train (Entrenamiento)')
    plt.plot(k_range, test_accuracies, marker='s', linestyle='-', color='crimson', linewidth=2.2, label='Accuracy en Test (Validación)')
    
    plt.axvline(best_k, color='forestgreen', linestyle=':', linewidth=2, label=f'K Óptimo = {best_k} (Accuracy = {best_acc:.4f})')
    plt.scatter([best_k], [best_acc], color='forestgreen', s=120, zorder=5)
    
    plt.title("Curva de Selección del Hiperparámetro K vs. Precisión (Ponderación por Distancia)", pad=12)
    plt.xlabel("Número de Vecinos Cercanos (K)")
    plt.ylabel("Precisión Global (Accuracy)")
    plt.xticks(k_range)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    ruta_grafica_k = os.path.join(carpeta_salida, "grafica_1_curva_k_accuracy.png")
    plt.savefig(ruta_grafica_k, dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_k, k_range, train_accuracies, test_accuracies, ruta_grafica_k

# ==============================================================================
# 3. ENTRENAMIENTO DEL MODELO FINAL Y MATRIZ DE CONFUSIÓN
# ==============================================================================

# Entrena el clasificador final con el K óptimo y ponderación por distancia.
def evaluar_modelo_final(best_k: int, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, 
                         y_train: pd.Series, y_test: pd.Series, carpeta_salida: str) -> tuple:
    clases_orden = ['Excelente', 'Buena', 'Regular / Proyecto']
    
    knn_final = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    knn_final.fit(X_train_scaled, y_train)
    y_pred = knn_final.predict(X_test_scaled)
    y_probs = knn_final.predict_proba(X_test_scaled)
    
    # Cálculo de métricas globales
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=clases_orden)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap de Matriz de Confusión
    plt.figure(figsize=(9, 7))
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"
            
    sns.heatmap(
        cm, 
        annot=annot_text, 
        fmt="", 
        cmap="Blues", 
        xticklabels=clases_orden, 
        yticklabels=clases_orden,
        cbar_kws={'label': 'Número de Observaciones Clasificadas'}
    )
    plt.title(f"Matriz de Confusión del Modelo KNN (K = {best_k}, Accuracy = {acc:.2%})", pad=12)
    plt.xlabel("Condición Predicha por el Modelo")
    plt.ylabel("Condición Real Observada")
    plt.tight_layout()
    
    ruta_grafica_cm = os.path.join(carpeta_salida, "grafica_2_matriz_confusion_knn.png")
    plt.savefig(ruta_grafica_cm, dpi=300, bbox_inches='tight')
    plt.close()
    
    return knn_final, y_pred, y_probs, cm, (acc, prec_macro, rec_macro, f1_macro), ruta_grafica_cm

# ==============================================================================
# 4. CASOS DE DESACUERDO CON ALTA CONFIANZA
# ==============================================================================

# Extrae ejemplos en el conjunto de prueba donde el modelo predijo una clase
# con alta probabilidad (> 85%) pero difiere de la etiqueta real.

def extraer_casos_desacuerdo(df_known: pd.DataFrame, idx_test: pd.Index, 
                             y_pred: np.ndarray, y_probs: np.ndarray, 
                             knn_model: KNeighborsClassifier) -> pd.DataFrame:
    test_df = df_known.loc[idx_test].copy()
    test_df['condicion_predicha'] = y_pred
    test_df['probabilidad_max'] = np.max(y_probs, axis=1)
    
    # Filtrar discrepancias
    discrepancias = test_df[test_df['target_condition'] != test_df['condicion_predicha']].copy()
    discrepancias_ordenadas = discrepancias.sort_values(by='probabilidad_max', ascending=False)
    
    cols_export = ['model', 'year', 'price', 'odometer', 'cylinders_num', 'target_condition', 'condicion_predicha', 'probabilidad_max']
    return discrepancias_ordenadas[cols_export]

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def main():
    ruta_dataset = os.path.join('Practica 1 - Limpieza de Datos', 'muscle_cars_clean.csv')
    carpeta_salida = os.path.join('Practica 6 - Clasificación de Datos', 'graficas')
    
    print("="*80)
    print("EJECUTANDO CLASIFICACIÓN DE DATOS CON KNN (PRÁCTICA 6)")
    print("="*80)
    
    df_known, X, y = cargar_y_preparar_datos(ruta_dataset)
    crear_directorio_graficas(carpeta_salida)
    print(f"Muestra con condición conocida: {len(df_known):,} observaciones (excluyendo 'unknown').")
    print(f"Distribución de la variable objetivo (3 clases):")
    print(y.value_counts(normalize=True).mul(100).round(2).to_string())
    print("\n" + "-"*80)
    
    # 1. División y Escalamiento
    print("\n>>> [1/4] Partición estratificada (80% train / 20% test) y escalamiento con StandardScaler...")
    X_train_s, X_test_s, y_train, y_test, idx_tr, idx_te, scaler = dividir_y_escalar_datos(X, y, df_known)
    print(f"    Conjunto de entrenamiento: {len(y_train):,} registros")
    print(f"    Conjunto de prueba (test):  {len(y_test):,} registros")
    
    # 2. Evaluación de K
    print("\n>>> [2/4] Evaluando hiperparámetro K de 1 a 25...")
    best_k, k_range, tr_acc, te_acc, r_graf_k = evaluar_curva_k(X_train_s, X_test_s, y_train, y_test, k_max=25, carpeta_salida=carpeta_salida)
    print(f"    K Óptimo seleccionado: K = {best_k} (Accuracy en Test = {max(te_acc):.4%})")
    print(f"    Gráfica guardada: {r_graf_k}")
    
    # 3. Evaluación del Modelo Final
    print(f"\n>>> [3/4] Evaluando Modelo Final (K = {best_k} con weights='distance')...")
    knn_mod, y_pred, y_probs, cm, metricas, r_graf_cm = evaluar_modelo_final(best_k, X_train_s, X_test_s, y_train, y_test, carpeta_salida)
    acc, prec, rec, f1 = metricas
    
    print("\nREPORTE DE CLASIFICACIÓN DETALLADO:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"    Gráfica de Matriz de Confusión guardada: {r_graf_cm}")
    print("\n" + "-"*80)
    
    # 4. Casos de Desacuerdo con Alta Confianza
    print("\n>>> [4/4] Extrayendo casos de desacuerdo con alta confianza (prob >= 0.85):")
    df_desacuerdos = extraer_casos_desacuerdo(df_known, idx_te, y_pred, y_probs, knn_mod)
    print(df_desacuerdos.head(8).to_string(index=False))
    
    print("\n" + "="*80)
    print("EJECUCIÓN COMPLETADA CON ÉXITO. TODAS LAS FIGURAS EXPORTADAS A 300 DPI.")
    print("="*80)

if __name__ == '__main__':
    main()
