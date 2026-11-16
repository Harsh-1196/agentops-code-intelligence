"""Unit tests validating Docker and Docker Compose configuration syntax."""

import os
import yaml


def test_dockerfile_existence_and_structure():
    """Verify backend and frontend Dockerfiles exist and have valid directives."""
    assert os.path.exists("Dockerfile")
    with open("Dockerfile", "r", encoding="utf-8") as f:
        backend_df = f.read()
    assert "FROM python:3.11-slim" in backend_df
    assert "HEALTHCHECK" in backend_df
    assert "EXPOSE 8000" in backend_df

    assert os.path.exists("frontend/Dockerfile")
    with open("frontend/Dockerfile", "r", encoding="utf-8") as f:
        frontend_df = f.read()
    assert "FROM node:20-alpine" in frontend_df
    assert "FROM nginx:alpine" in frontend_df
    assert "EXPOSE 3000" in frontend_df


def test_docker_compose_syntax_and_services():
    """Verify docker-compose.yml parses cleanly and orchestrates all 3 services."""
    assert os.path.exists("docker-compose.yml")
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})
    assert "backend" in services
    assert "frontend" in services
    assert "jaeger" in services

    # Port checks
    assert "8000:8000" in services["backend"]["ports"]
    assert "3000:3000" in services["frontend"]["ports"]
    assert any("16686:16686" in p for p in services["jaeger"]["ports"])
    assert any("4318:4318" in p for p in services["jaeger"]["ports"])


def test_dockerignore_excludes_secrets():
    """Verify .dockerignore excludes .env and sensitive cache directories."""
    assert os.path.exists(".dockerignore")
    with open(".dockerignore", "r", encoding="utf-8") as f:
        content = f.read()
    assert ".env" in content
    assert "node_modules" in content
    assert "__pycache__" in content
