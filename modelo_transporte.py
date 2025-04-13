import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Crear dataset simulado
np.random.seed(0)
n = 20  
data = pd.DataFrame({
    'origen': np.random.choice(['Est. A', 'Est. B', 'Est. C'], size=n),
    'destino': np.random.choice(['Est. D', 'Est. E', 'Est. F'], size=n),
    'hora': np.random.randint(5, 23, size=n),
    'dia_semana': np.random.choice(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes'], size=n),
    'clima': np.random.choice(['Soleado', 'Lluvia', 'Nublado'], size=n),
    'tiempo_viaje': np.random.normal(loc=15, scale=3, size=n).round(2)
})

# Guardar CSV
data.to_csv("dataset_transporte_masivo.csv", index=False)

# Preprocesamiento
le_origen = LabelEncoder()
le_destino = LabelEncoder()
le_dia = LabelEncoder()
le_clima = LabelEncoder()

data['origen'] = le_origen.fit_transform(data['origen'])
data['destino'] = le_destino.fit_transform(data['destino'])
data['dia_semana'] = le_dia.fit_transform(data['dia_semana'])
data['clima'] = le_clima.fit_transform(data['clima'])

# Separar variables predictoras y objetivo
X = data.drop('tiempo_viaje', axis=1)
y = data['tiempo_viaje']

# Entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
error = mean_squared_error(y_test, y_pred) ** 0.5
print(f"Error promedio (RMSE): {error:.2f} minutos")
