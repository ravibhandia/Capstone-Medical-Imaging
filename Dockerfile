FROM python:3.7-slim
WORKDIR /deploy

COPY ./requirements.txt /deploy
COPY ./app.py /deploy
COPY ./inference.py /deploy
COPY weights /deploy/weights
COPY templates /deploy/templates
COPY static /deploy/static

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python","app.py"]