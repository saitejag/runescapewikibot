FROM python:3.13-slim
# Install uv
RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

COPY src/ src/

RUN uv pip install --system -e .

COPY . .

EXPOSE 8000

CMD ["python","src/rswiki_bot_api.py"]


