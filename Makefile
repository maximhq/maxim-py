.PHONY: build publish docs docs-clean docs-serve

build:
	rm -rf dist build maxim_py.egg-info
	python3 -m build

publish:
	rm -rf dist build maxim_py.egg-info
	. .venv/bin/activate && \
	python3 -m build && \
	twine upload dist/*

# Documentation commands
docs: docs-clean
	@echo "Generating documentation..."
	pydoc-markdown -I . -p maxim --render-toc > maxim_full.md
	python3 scripts/split_docs.py maxim_full.md docs/
	@echo "Documentation generated in docs/ directory"

docs-clean:
	@echo "Cleaning documentation..."
	rm -rf docs/
	rm -f maxim_full.md maxim.md maxim.mdx
	@echo "Documentation cleaned"

docs-serve: docs
	@echo "Starting documentation server..."
	@if command -v python3 -m http.server >/dev/null 2>&1; then \
		cd docs && python3 -m http.server 8000; \
	else \
		cd docs && python -m SimpleHTTPServer 8000; \
	fi