.PHONY: test frontier outliers clean

test:
	uv run pytest -q

frontier:
	uv run python bench/frontier.py

outliers:
	uv run python bench/outliers.py

clean:
	rm -rf .pytest_cache **/__pycache__ results/tinygpt.pt
