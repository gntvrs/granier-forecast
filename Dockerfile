FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Por defecto no ejecuta nada: lo hará el comando del Job
CMD ["python", "main.py"]
