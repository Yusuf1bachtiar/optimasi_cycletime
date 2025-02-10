FROM python:3.9-slim-buster  # Or a later version if needed

WORKDIR /app

COPY requirements.txt .

# Install system dependencies if needed (often necessary for MySQL connector)
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev default-libmysqlclient-dev

RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install -r requirements.txt

COPY . .

# IMPORTANT: Choose ONE of the following CMD instructions, depending on how you structure your app

# Option 1: If your Flask app is in a file named app.py and the app instance is named app:
# CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"] 

# Option 2: If your Flask app is in a file named main.py and the app instance is named app:
CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "main:app"]

# Option 3: If your Flask app is in a file named my_web_app.py and the app instance is named application:
# CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "my_web_app:application"]

# Option 4: If you have a wsgi.py file and you create the Flask app in it:
# CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "wsgi"]

# Option 5: If you are using Flask blueprints and you have a create_app function:
# CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "wsgi:create_app()"] # where wsgi.py contains the function create_app()

# Option 6: If your entrypoint is in run.py
# CMD ["/opt/venv/bin/gunicorn", "--bind", "0.0.0.0:$PORT", "run:app"]
