# Automatically detect host UID/GID
UID := $(shell id -u)
GID := $(shell id -g)

export HOST_UID := $(UID)
export HOST_GID := $(GID)

.PHONY: build
build:
	docker compose build

.PHONY: run
run:
	docker compose run --rm project

.PHONY: dev
dev:
	docker compose run --rm dev

.PHONY: up
up:
	docker compose up -d --remove-orphans

.PHONY: down
down:
	docker compose down --remove-orphans

.PHONY: prune
prune:
	docker system prune -f
	docker volume prune -f
