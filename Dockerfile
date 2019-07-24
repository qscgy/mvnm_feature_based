FROM python:latest

RUN apt-get install git -y
RUN pip install numpy scikit-learn opencv-python h5py scikit-image

ENTRYPOINT [ "/bin/bash" ]