FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Install basic packages
RUN apt update
RUN apt install gnupg git curl make g++ pip -y
