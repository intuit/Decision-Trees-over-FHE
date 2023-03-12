FROM ubuntu:20.04

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND="noninteractive" TZ="America/Los_Angeles" apt-get install -y git zip wget cmake g++

WORKDIR /usr/src/myapp
RUN git clone --branch 4.0.0 https://github.com/Microsoft/SEAL.git
WORKDIR /usr/src/myapp/SEAL
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build
RUN cmake --install build

COPY . /usr/src/myapp
WORKDIR /usr/src/myapp
RUN cmake -S . -B build
RUN cmake --build build
WORKDIR build
CMD ./fhe_random_forest


