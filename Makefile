.PHONY: build publish

build:
	rm -rf dist build maxim_py.egg-info
	python3 -m build

publish:
	rm -rf dist build maxim_py.egg-info
	. .venv/bin/activate && \
	python3 -m build && \
	twine upload dist/*