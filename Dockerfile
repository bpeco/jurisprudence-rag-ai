FROM python:3.12
WORKDIR /app

# 1) Copia deps y código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 2) Expón el puerto FastAPI
EXPOSE 8080

# 3) Arranca Uvicorn apuntando a main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
