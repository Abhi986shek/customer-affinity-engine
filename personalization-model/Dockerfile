FROM python:3.11-slim

WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/opt/ml/code
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]
