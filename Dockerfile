FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev default-libmysqlclient-dev

RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install -r requirements.txt

COPY check_venv.py /app/
RUN chmod +x /app/check_venv.py # Opsional, jika ada masalah izin

RUN /bin/bash -c "source /opt/venv/bin/activate && /opt/venv/bin/python /app/check_venv.py" # Jalankan dalam satu baris

COPY . .

CMD ["/bin/bash", "-c", "source /opt/venv/bin/activate && /opt/venv/bin/gunicorn --bind 0.0.0.0:$PORT app_main:app"]
