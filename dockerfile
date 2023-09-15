# Use an official Python runtime as the base image
FROM python:3.8-slim

#install ffmpeg
RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip install --upgrade pip

# Copy the FastAPI service files into the container's working directory
COPY requirements.txt requirements.txt

# Install necessary packages
RUN pip install -r requirements.txt

# get arguments for private repo
ARG ARIN_PYPI_REPOSITORY_URL
ARG ARIN_PYPI_USERNAME
ARG ARIN_PYPI_PASSWORD

#TODO this next line might need some work
RUN echo "[global]\nindex-url = $ARIN_PYPI_REPOSITORY_URL\ntrusted-host = your.private.pypi.repo\nextra-index-url = https://pypi.org/simple\nusername = $ARIN_PYPI_USERNAME\npassword = $ARIN_PYPI_PASSWORD" > /etc/pip.conf

COPY /module_name.txt /module_name.txt

COPY /arin_text_classifier /arin_text_classifier/
COPY /setup.py /setup.py
COPY /README.md /README.md

# Install the app library
RUN pip install -e .

# Copy the FastAPI service files into the container's working directory
COPY /app /app/

# Set the working directory inside the container
WORKDIR /app

# Expose port 8000 for the FastAPI service
EXPOSE 8000

# Command to run the FastAPI service with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
