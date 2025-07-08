# WeatherServiceAPI

This project provides a RESTful API for forecasting weather conditions using LSTM model.  
The API is built with **FastAPI** and packaged in a **Docker** container for easy deployment.

## Features

- Predicts predicts maximum temperature for upcoming days.
- Lightweight and containerized for rapid deployment in any environment.

## Installation

Clone the repository:

```bash
git clone https://github.com/FedorKaltiugin/WeatherServiceAPI.git

cd WeatherServiceAPI

Build the Docker image:

docker build -t weather-service .

Run the container:

docker run -d --name weather-s -p 5000:8000 weather-service

After that, the API will be available at:

http://localhost:5000/docs (swagger UI for testing endpoints)
