FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev default-libmysqlclient-dev # Jika diperlukan

RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install -r requirements.txt

COPY . .

CMD ["/opt/venv/bin/python", "app_main.py"]  # Ganti app_main.py dengan nama file aplikasi Anda
