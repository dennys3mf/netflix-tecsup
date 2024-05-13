# Usar una imagen base de Python
FROM python:3.8-slim-buster

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos primero para aprovechar la caché de capas de Docker
COPY requirements.txt .

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente del proyecto al contenedor
COPY . .

# Exponer el puerto en el que se ejecutará la aplicación Flask
EXPOSE 80

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]