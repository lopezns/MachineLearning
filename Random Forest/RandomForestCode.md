# Explicación del Código de Clasificación con Random Forest

Este documento describe un código de clasificación utilizando el algoritmo Random Forest. A continuación se presenta la explicación de cada sección del código.

## 1. Importar las Bibliotecas Necesarias

>> ```python
>> import numpy as np
>> import pandas as pd
>> import matplotlib.pyplot as plt
>> from sklearn.model_selection import train_test_split
>> from sklearn.ensemble import RandomForestClassifier
>> from sklearn.tree import export_graphviz
>> from sklearn import tree
>> import pydot
>> from io import StringIO
>> from IPython.display import Image
>> from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
>> !pip install category_encoders
>> import category_encoders as ce
>> ```

- **NumPy**: Utilizado para realizar operaciones numéricas.
- **Pandas**: Utilizado para manipulación de datos en forma de DataFrame.
- **Matplotlib**: Utilizado para crear visualizaciones.
- **Scikit-learn**: Proporciona herramientas para modelos de machine learning, métricas, y visualización de árboles.
- **Category Encoders**: Permite convertir variables categóricas en formato numérico.

## 2. Cargar el Dataset

>> ```python
>> dat = pd.read_csv('car_evaluation.csv')
>> print(dat.head())  # Ver las primeras filas para identificar las columnas
>> ```

- **`pd.read_csv()`**: Carga el archivo CSV en un DataFrame.
- **`print(dat.head())`**: Muestra las primeras filas del DataFrame para verificar las columnas.

## 3. Convertir Variables Categóricas a Numéricas

>> ```python
>> encoder = ce.OneHotEncoder(use_cat_names=True)
>> dat_encoded = encoder.fit_transform(dat)
>> ```

- **`ce.OneHotEncoder()`**: Inicializa el codificador One-Hot para convertir variables categóricas en columnas binarias.
- **`fit_transform()`**: Aplica el codificador y transforma los datos.

## 4. Dividir el Dataset en Variables Independientes y Dependientes

>> ```python
>> X = dat_encoded.iloc[:, :-1]  # Todas las columnas excepto la última
>> y = dat_encoded.iloc[:, -1]   # Última columna, asumiendo que esta es la etiqueta
>> ```

- **`X`**: Contiene todas las columnas excepto la última, que son las características.
- **`y`**: Contiene la última columna, que es la variable objetivo (etiqueta).

## 5. Dividir los Datos en Conjuntos de Entrenamiento y Prueba

>> ```python
>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
>> ```

- **`train_test_split()`**: Divide los datos en conjuntos de entrenamiento (70%) y prueba (30%).

## 6. Entrenamiento del Modelo Random Forest

### 6.1 Inicializar el Modelo

>> ```python
>> rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
>> ```

- **`RandomForestClassifier()`**: Crea un modelo de Random Forest con 10 árboles.

### 6.2 Entrenar el Modelo

>> ```python
>> rf_model.fit(X_train, y_train)
>> ```

- **`fit()`**: Entrena el modelo con los datos de entrenamiento.

### 6.3 Hacer Predicciones

>> ```python
>> y_pred = rf_model.predict(X_test)
>> ```

- **`predict()`**: Realiza predicciones utilizando los datos de prueba.

### 6.4 Obtener Predicciones de Cada Árbol

>> ```python
>> tree_predictions = np.array([tree.predict(X_test) for tree in rf_model.estimators_])
>> ```

- Se generan predicciones individuales para cada árbol en el modelo.

### 6.5 Mostrar la Votación de Cada Árbol

>> ```python
>> print("\nPredicciones de cada árbol:")
>> for i, pred in enumerate(tree_predictions.T):  # Transpuesta para tener las predicciones por fila
>>     print(f"Instancia {i + 1}: Votos = {pred}, Clase Final = {y_pred[i]}")
>>     
>>     # Conteo de los votos por clase
>>     unique, counts = np.unique(pred, return_counts=True)
>>     vote_count = dict(zip(unique, counts))
>> 
>>     print(f"Votación mayoritaria: {vote_count} -> Clase final: {y_pred[i]}")
>> ```

- Se muestra la votación de cada árbol y la clase final elegida.

## 7. Evaluación del Modelo

### 7.1 Calcular la Precisión

>> ```python
>> accuracy = accuracy_score(y_test, y_pred)
>> print(f"\nAccuracy: {accuracy * 100:.2f}%")
>> ```

- **`accuracy_score()`**: Calcula la precisión del modelo.

### 7.2 Mostrar la Matriz de Confusión

>> ```python
>> conf_matrix = confusion_matrix(y_test, y_pred)
>> print("\nConfusion Matrix:")
>> print(conf_matrix)
>> ```

- **`confusion_matrix()`**: Genera la matriz de confusión para evaluar el rendimiento del modelo.

### 7.3 Mostrar el Reporte de Clasificación

>> ```python
>> class_report = classification_report(y_test, y_pred)
>> print("\nClassification Report:")
>> print(class_report)
>> ```

- **`classification_report()`**: Proporciona estadísticas sobre la precisión, recall y F1-score del modelo.

## 8. Visualización de un Árbol de Decisión del Random Forest

### 8.1 Seleccionar un Árbol del Bosque

>> ```python
>> tree_num = 0
>> selected_tree = rf_model.estimators_[tree_num]
>> ```

- Se selecciona el primer árbol del modelo.

### 8.2 Visualización del Árbol

>> ```python
>> plt.figure(figsize=(20,10))
>> tree.plot_tree(selected_tree, filled=True, rounded=True)
>> plt.show()
>> ```

- **`plot_tree()`**: Dibuja el árbol seleccionado utilizando Matplotlib.

## 9. Exportar y Mostrar el Árbol en Formato Gráfico

>> ```python
>> dot_data = StringIO()
>> export_graphviz(selected_tree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
>> (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
>> 
>> # Mostrar el árbol como imagen
>> Image(graph.create_png())
>> ```

- **`export_graphviz()`**: Exporta el árbol a formato DOT.
- **`pydot.graph_from_dot_data()`**: Convierte el DOT a un gráfico.
- **`Image()`**: Muestra el árbol como imagen.

## 10. Visualización de la Matriz de Confusión

>> ```python
>> import numpy as np
>> import matplotlib.pyplot as plt
>> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
>> 
>> # Clases reales (usando y_test, que contiene las clases reales del conjunto de prueba)
>> y_true = y_test.to_numpy()
>> 
>> # Clases predichas por tu modelo (ya tenemos y_pred)
>> y_pred = y_pred
>> 
>> # Generar matriz de confusión
>> cm = confusion_matrix(y_true, y_pred)
>> 
>> # Mostrar la matriz de confusión de forma visual
>> disp = ConfusionMatrixDisplay(confusion_matrix=cm)
>> disp.plot()
>> 
>> # Ajustes visuales para hacerla más clara
>> plt.title("Matriz de Confusión")
>> plt.xlabel("Clases Predichas")
>> plt.ylabel("Clases Reales")
>> plt.show()
>> ```

- Se genera y visualiza la matriz de confusión de forma clara y comprensible.
