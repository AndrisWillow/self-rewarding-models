FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /usr/src/app

# Installing necessary packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Keep container running
CMD ["tail", "-f", "/dev/null"]
