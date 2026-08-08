.PHONY: up down logs ps reset

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

reset:
	docker compose down -v
