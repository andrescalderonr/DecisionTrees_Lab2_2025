import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

# Cargar dataset
base_path = os.path.join(os.path.dirname(__file__), '../../resources')
train_path = os.path.join(base_path, 'adult.data')
test_path = os.path.join(base_path, 'adult.test')

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

train_data = pd.read_csv(train_path, names=columns, na_values='?', skipinitialspace=True)
test_data = pd.read_csv(test_path, names=columns, na_values='?', skipinitialspace=True, skiprows=1)

data = pd.concat([train_data, test_data], axis=0)
data['income'] = data['income'].str.replace('.', '', regex=False).str.strip()

# Reemplazar valores nulos por la moda de cada columna
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])

# Codificación de variables categóricas
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separar variables predictoras (X) y objetivo (y)
X = data.drop('income', axis=1)
y = data['income']

# Normalización / estandarización solo de las variables numéricas de X
scaler = StandardScaler()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[num_cols] = scaler.fit_transform(X[num_cols])

# División de datos: 70% entrenamiento, 10% validación, 20% prueba
# Primero 70% entrenamiento y 30% temporal (validación + prueba)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# De los 30% restantes: validación (10%) y prueba (20%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp)

print("Tamaño del conjunto de entrenamiento:", X_train.shape)
print("Tamaño del conjunto de validación:", X_val.shape)
print("Tamaño del conjunto de prueba:", X_test.shape)

# Paso 3.1.1: Entrenar un Árbol de Decisión

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_val)
f1_tree = f1_score(y_val, y_pred_tree)

print("F1-score (Árbol de Decisión):", round(f1_tree, 4))

# Paso 3.1.2: Entrenar un Bosque Aleatorio

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)
f1_rf = f1_score(y_val, y_pred_rf)

print("F1-score (Bosque Aleatorio):", round(f1_rf, 4))

# Paso 3.1.3: Entrenar un Bosque con Potenciación de Gradiente

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_val)
f1_gb = f1_score(y_val, y_pred_gb)

print("F1-score (Gradient Boosting):", round(f1_gb, 4))

# Comparar los tres F1-scores
scores = {
    'Decision Tree': f1_tree,
    'Random Forest': f1_rf,
    'Gradient Boosting': f1_gb
}

best_model_name = max(scores, key=scores.get)
print("\nMejor modelo según F1 en validación:", best_model_name)

# Seleccionar el modelo ganador
if best_model_name == 'Decision Tree':
    best_model = tree_model
elif best_model_name == 'Random Forest':
    best_model = rf_model
else:
    best_model = gb_model

# Evaluar con el conjunto de prueba
y_pred_test = best_model.predict(X_test)
f1_test = f1_score(y_test, y_pred_test)

print("\nF1-score final en el conjunto de prueba:", round(f1_test, 4))



# Operaciones para hcer conclusiones del paso 4

# Altura (profundidad del árbol)
altura = tree_model.get_depth()

# Número de reglas (nodos)
n_reglas = tree_model.tree_.node_count

# Impureza promedio (Gini en cada nodo)
impureza_promedio = np.mean(tree_model.tree_.impurity)

# Exactitud en entrenamiento y validación
acc_train = tree_model.score(X_train, y_train)
acc_val = tree_model.score(X_val, y_val)

print("\nÁrbol de Decisión:")
print(f" - Altura: {altura}")
print(f" - N° de reglas: {n_reglas}")
print(f" - Impureza promedio: {impureza_promedio:.4f}")
print(f" - Exactitud entrenamiento: {acc_train:.4f}")
print(f" - Exactitud validación: {acc_val:.4f}")



# Profundidad promedio de los árboles
alturas = [estimator.get_depth() for estimator in rf_model.estimators_]
altura_promedio = np.mean(alturas)

# Número promedio de nodos
reglas = [estimator.tree_.node_count for estimator in rf_model.estimators_]
reglas_promedio = np.mean(reglas)

# Impureza promedio entre todos los árboles
impurezas = [np.mean(estimator.tree_.impurity) for estimator in rf_model.estimators_]
impureza_promedio = np.mean(impurezas)

acc_train = rf_model.score(X_train, y_train)
acc_val = rf_model.score(X_val, y_val)

print("\nBosque Aleatorio:")
print(f" - Altura promedio: {altura_promedio:.2f}")
print(f" - N° promedio de reglas: {reglas_promedio:.2f}")
print(f" - Impureza promedio: {impureza_promedio:.4f}")
print(f" - Exactitud entrenamiento: {acc_train:.4f}")
print(f" - Exactitud validación: {acc_val:.4f}")


# Profundidad promedio de los árboles
alturas = [estimator[0].get_depth() for estimator in gb_model.estimators_]
altura_promedio = np.mean(alturas)

# N° promedio de reglas
reglas = [estimator[0].tree_.node_count for estimator in gb_model.estimators_]
reglas_promedio = np.mean(reglas)

# Impureza promedio
impurezas = [np.mean(estimator[0].tree_.impurity) for estimator in gb_model.estimators_]
impureza_promedio = np.mean(impurezas)

acc_train = gb_model.score(X_train, y_train)
acc_val = gb_model.score(X_val, y_val)

print("\nGradient Boosting:")
print(f" - Altura promedio: {altura_promedio:.2f}")
print(f" - N° promedio de reglas: {reglas_promedio:.2f}")
print(f" - Impureza promedio: {impureza_promedio:.4f}")
print(f" - Exactitud entrenamiento: {acc_train:.4f}")
print(f" - Exactitud validación: {acc_val:.4f}")