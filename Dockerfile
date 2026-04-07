FROM python:3.11-slim AS base

ENV TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

RUN pip install --upgrade pip && pip install --no-cache-dir gendercast

COPY . .

CMD ["bash", "run.sh"]
