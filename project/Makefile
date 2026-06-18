.PHONY: test train run docker-run

test:
	pytest tests/

train:
	python -m src.models.train

run:
	python -m src.service

docker-run:
	docker-compose up --build