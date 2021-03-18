FROM tensorflow/tensorflow:latest-gpu

RUN echo "\nTensorflow latest gpu base image downloaded...\n"


# copy the files from the current server directory into the docker.
WORKDIR /workspace/
ADD . /workspace/
RUN pip install -r requirements.txt

# open cv error
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install vim nano  -y