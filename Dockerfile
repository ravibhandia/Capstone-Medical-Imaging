FROM python:3.7-alpine
WORKDIR /deploy
ADD . /deploy
RUN pip install -r requirements.txt
CMD ["python","app.py"]