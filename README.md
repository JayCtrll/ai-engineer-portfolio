# AI Engineer Portfolio - FastAPI Foundations

A production-ready learning project showcasing FastAPI best practices.

## Features

- ✅ Health check endpoint
- ✅ Request/response validation with Pydantic
- ✅ Logging middleware
- ✅ Bearer token authentication
- ✅ Sync vs async performance comparison
- 🐳 Docker support

## Tech Stack

Python, FastAPI, Pydantic, Uvicorn, Docker, WSL2

## Project Structure Tree

ai-engineer-portfolio/  

├── app/  

│ ├── init.py  

│ ├── main.py # FastAPI entry, all API routes & OpenAPI config  

│ ├── middleware.py # Global HTTP logging middleware implementation  

│ ├── auth.py # Token verification dependency logic  

│ └── slow.py # Synchronous / asynchronous slow task test functions  

├── .gitignore # Git ignore rules for venv, cache, logs, build artifacts  

├── .python-version # Lock target Python interpreter version  

├── requirements.txt # Python project dependency list  

├── Dockerfile # Docker container build definition  

└── README.md # Project documentation

## Quick Start Guide
### Method : Run via Docker Container
1.Build docker image  

`docker build -t fastapi-week1:latest .`  

2.Start container service  

`docker run -d -p 8000:8000 --name fastapi-app fastapi-week1:latest`
