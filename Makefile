.PHONY: install test run deploy clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src

run:
	uvicorn api.main:app --reload

deploy:
	docker-compose up --build -d 

logs:
	docker-compose logs -f 

clean:
	docker-compose down -v
