import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Tamaño de casas (m²)
tamano = np.array([50, 80, 120, 150, 200]).reshape(-1, 1)

# Precios (miles de €)
precio = np.array([150, 230, 350, 420, 580])

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(tamano, precio)

# Coeficientes aprendidos
print(f"Pendiente: {modelo.coef_[0]:.2f}")
print(f"Intercepto: {modelo.intercept_:.2f}")

# Predecir precio de casa de 100 m²
nueva_casa = np.array([[100]])
prediccion = modelo.predict(nueva_casa)
print(f"Precio estimado: {prediccion[0]:.0f}k €")

# Visualizar datos y modelo
plt.scatter(tamano, precio,
            color="#9CEC14",
            label='Datos reales')

# Línea de predicción
plt.plot(tamano,
         modelo.predict(tamano),
         color="#9B098A",
         linewidth=2,
         label='Modelo entrenado')

plt.xlabel('Tamaño (m²)')
plt.ylabel('Precio (miles €)')
plt.legend()
plt.title('Predicción de precios')
plt.savefig("imagen.png", dpi=300, bbox_inches='tight')
plt.show()