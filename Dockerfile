FROM python:3.8

RUN mkdir -p /app/medical_nn_service
WORKDIR /app/medical_nn_service

COPY . /app/medical_nn_service

EXPOSE 8080

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "src.service.app"]
