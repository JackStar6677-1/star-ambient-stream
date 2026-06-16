FROM python:3.12-slim

RUN apt-get update && apt-get install -y ffmpeg sqlite3 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir     google-api-python-client google-auth-oauthlib google-auth-httplib2     numpy scipy pedalboard

COPY . .

CMD ["python", "-u", "stream_manager.py"]
