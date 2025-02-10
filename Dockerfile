FROM python:3

# Run in unbuffered mode
ENV PYTHONUNBUFFERED=1 

WORKDIR /app

COPY . .

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev default-libmysqlclient-dev

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "app_main:app"]
