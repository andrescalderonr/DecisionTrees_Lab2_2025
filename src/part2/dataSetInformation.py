import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



# Obtener ruta absoluta a la carpeta resources
base_path = os.path.join(os.path.dirname(__file__), '../../resources')

train_path = os.path.join(base_path, 'adult.data')
test_path = os.path.join(base_path, 'adult.test')

# Definir nombres de columnas
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Cargar archivos
train_data = pd.read_csv(train_path, names=columns, na_values='?', skipinitialspace=True)
test_data = pd.read_csv(test_path, names=columns, na_values='?', skipinitialspace=True, skiprows=1)

# Unir para análisis conjunto
data = pd.concat([train_data, test_data], axis=0)

# Normalizar la columna 'income'
data['income'] = data['income'].str.replace('.', '', regex=False).str.strip()

print("Dimensiones del dataset:", data.shape)
print("\nPrimeras filas del dataset:")
print(data.head())

print("\nValores nulos por columna:")
print(data.isnull().sum())

print("\nDistribución de la variable objetivo (income):")
print(data['income'].value_counts())

# Seleccionar solo columnas numéricas
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Calcular matriz de correlación
corr_matrix = numeric_data.corr()

# Mostrar la matriz completa
print("\nMatriz de correlación:")
print(corr_matrix)

# Visualización con mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación entre variables numéricas")
plt.show()




