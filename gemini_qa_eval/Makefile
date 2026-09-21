# src/ layout: either `pip install -e .` or keep this on the path.
export PYTHONPATH := src

.PHONY: install test demo answer score report workbook agreement clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests -q

# Full pipeline with no API key and no network - proves the wiring works.
demo:
	GQE_MOCK=1 RETRIEVAL_STRATEGY=bm25 python -m gqe.cli all --limit 3 --run-id demo -v

answer:
	python -m gqe.cli answer -v

score:
	python -m gqe.cli score -v

report:
	python -m gqe.cli report

workbook:
	python -m gqe.cli workbook --reviewers reviewer_1,reviewer_2

clean:
	rm -rf .cache runs/demo runs/smoke
