FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev default-libmysqlclient-dev # Jika diperlukan untuk MySQL atau dependensi lain

RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install -r requirements.txt

COPY . .

CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "app_main:app"]
