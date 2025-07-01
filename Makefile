.PHONY: build publish docs docs-clean docs-serve

build:
	rm -rf dist build maxim_py.egg-info
	python3 -m build

publish:
	rm -rf dist build maxim_py.egg-info
	. .venv/bin/activate && \
	python3 -m build && \
	twine upload dist/*
	@echo "Package published successfully, creating GitHub release..."
	@VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Creating tag and release for version: $$VERSION"; \
	git tag -a "v$$VERSION" -m "Release v$$VERSION" || echo "Tag already exists"; \
	git push origin "v$$VERSION" || echo "Tag already pushed"; \
	echo "import re, sys" > /tmp/extract_notes.py; \
	echo "with open('README.md', 'r') as f: content = f.read()" >> /tmp/extract_notes.py; \
	echo "version = sys.argv[1]" >> /tmp/extract_notes.py; \
	echo "pattern = f'### {version}\\\\n([\\\\s\\\\S]*?)(?=\\\\n### |\\\\n\\\\n##|$$)'" >> /tmp/extract_notes.py; \
	echo "match = re.search(pattern, content)" >> /tmp/extract_notes.py; \
	echo "notes = match.group(1).strip() if match else f'Release notes for version {version}'" >> /tmp/extract_notes.py; \
	echo "print(notes)" >> /tmp/extract_notes.py; \
	python3 /tmp/extract_notes.py $$VERSION > /tmp/release_notes.txt; \
	gh release create "v$$VERSION" \
		--title "Release v$$VERSION" \
		--notes-file /tmp/release_notes.txt \
		--verify-tag || echo "Release creation failed or already exists"; \
	rm -f /tmp/release_notes.txt /tmp/extract_notes.py

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