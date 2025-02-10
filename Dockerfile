# Use a specific Python version
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev default-libmysqlclient-dev

RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install -r requirements.txt

COPY check_venv.py /app/
RUN /bin/bash -c "source /opt/venv/bin/activate && python /app/check_venv.py"

RUN /bin/bash -c "source /opt/venv/bin/activate && which gunicorn" # Periksa jalur gunicorn
RUN /bin/bash -c "source /opt/venv/bin/activate && which python" # Periksa jalur python

COPY . .

CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "app_main:app"]  # Corrected CMD instruction with double quotes
