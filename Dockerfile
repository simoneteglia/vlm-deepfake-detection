FROM docker.io/pytorch/pytorch:latest

WORKDIR /workspace

COPY . /workspace

RUN apt-get update && apt-get install -y sudo

RUN pip3 install -r requirements.txt

CMD ["/bin/sh", "-c", "bash"]