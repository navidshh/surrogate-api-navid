FROM tensorflow/tensorflow:2.8.0

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN python -m pip install --no-cache-dir --upgrade pip setuptools

WORKDIR /home/btap_ml

COPY requirements.txt ./
COPY src ./src

RUN pip install setuptools==65.5.1 packaging==21.3
RUN pip install -r requirements.txt

RUN mkdir output
RUN mkdir input

COPY input ./input

WORKDIR /home/btap_ml/src

RUN pip3 install "pydantic[dotenv]"
RUN pip3 install uvicorn
RUN pip3 install python-multipart
RUN pip3 install pyjwt[crypto]
RUN pip3 install psutil


CMD ["/bin/bash"]

