#!/usr/bin/env python3
"""
Split pydoc-markdown generated documentation into individual module files.
"""
import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def clean_title_text(text: str) -> str:
    """Clean title text by removing markdown links and other unwanted formatting."""
    # Remove markdown links [text](url) and keep just the text
    import re

    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove any remaining markdown formatting
    text = text.replace("**", "").replace("*", "").replace("`", "")
    return text.strip()


def clean_content(
    content: str, all_definitions: Dict[str, Dict[str, str]] = None
) -> str:
    """Clean the content by removing anchor tags and other unwanted elements."""
    # Remove anchor tags like <a id="..."></a>
    content = re.sub(r'<a id="[^"]*"></a>\s*', "", content)

    # Remove " Objects" suffix from class headers
    content = re.sub(
        r"^## ([A-Za-z_][A-Za-z0-9_]*) Objects$", r"## \1", content, flags=re.MULTILINE
    )

    # Clean up ALL headers to remove unwanted markdown links
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#"):
            # Extract just the text content from ANY header, removing all markdown links
            title_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
            lines[i] = title_text
    content = "\n".join(lines)

    # Remove empty lines at the beginning
    content = content.lstrip("\n")

    # Convert attribute lists to tables with internal links for custom types
    content = convert_attributes_to_table(content, all_definitions)

    return content


def find_class_and_function_definitions(content: str) -> Dict[str, Dict[str, str]]:
    """Find all class and function definitions in the content with their module paths."""
    definitions = {}
    lines = content.split("\n")
    current_module = None

    for line in lines:
        # Track current module
        if line.startswith("# maxim."):
            current_module = line[2:].strip()  # Remove "# " prefix
            continue

        # Find class definitions (## ClassName Objects)
        class_match = re.match(r"^## ([A-Za-z_][A-Za-z0-9_]*) Objects", line)
        if class_match and current_module:
            class_name = class_match.group(1)
            definitions[class_name] = {
                "type": "class",
                "module": current_module,
                "path": create_docs_json_path(current_module),
            }
            continue

        # Find function definitions (#### function_name)
        func_match = re.match(r"^#### ([a-z_\\][a-z0-9_\\]*)", line)
        if func_match and current_module:
            func_name = func_match.group(1)
            definitions[func_name] = {
                "type": "function",
                "module": current_module,
                "path": create_docs_json_path(current_module),
            }
            continue

    return definitions


def find_line_number_in_file(file_path: str, object_name: str, object_type: str) -> int:
    """Find the line number where a class or function is defined in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Clean the object name (remove escape characters from pydoc-markdown)
        clean_object_name = object_name.replace("\\_", "_")

        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()

            if object_type == "class":
                # Look for class definition
                if stripped_line.startswith(
                    f"class {clean_object_name}("
                ) or stripped_line.startswith(f"class {clean_object_name}:"):
                    return i
            elif object_type == "function":
                # Look for function definition (including async functions)
                if stripped_line.startswith(
                    f"def {clean_object_name}("
                ) or stripped_line.startswith(f"async def {clean_object_name}("):
                    return i
                # Also check for methods inside classes
                elif stripped_line.startswith(
                    f"    def {clean_object_name}("
                ) or stripped_line.startswith(f"    async def {clean_object_name}("):
                    return i

        # If not found, return 1 as fallback
        return 1

    except (FileNotFoundError, UnicodeDecodeError, Exception):
        # If file can't be read, return 1 as fallback
        return 1


def get_github_source_link(
    module_name: str, object_name: str = None, object_type: str = None
) -> str:
    """Generate GitHub source link for a module or specific object."""
    base_url = "https://github.com/maximhq/maxim-py/blob/main"

    # Clean the module name (remove escape characters from pydoc-markdown)
    clean_module_name = module_name.replace("\\_", "_")

    # Convert module name to file path
    module_path = clean_module_name.replace(".", "/")

    # Remove 'maxim' prefix since it's the root package
    if module_path.startswith("maxim/"):
        module_path = module_path[6:]
    elif module_path == "maxim":
        module_path = ""

    # Construct file path
    if module_path:
        file_path = f"maxim/{module_path}.py"
    else:
        file_path = "maxim/__init__.py"

    github_link = f"{base_url}/{file_path}"

    # Add anchor for specific object if provided
    if object_name and object_type:
        # Find the actual line number in the source file
        line_number = find_line_number_in_file(file_path, object_name, object_type)
        github_link += f"#L{line_number}"

    return github_link


def add_github_source_links(
    content: str, all_definitions: Dict[str, Dict[str, str]]
) -> str:
    """Add GitHub source links to class and function headers."""
    lines = content.split("\n")
    result_lines = []
    current_module = None

    in_code_block = False

    for i, line in enumerate(lines):
        # Track if we're inside a code block
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result_lines.append(line)
            continue

        # Track current module
        if line.startswith("# maxim."):
            current_module = line[2:].strip()
            result_lines.append(line)
            continue

        # Add GitHub source link after class/function headers
        # Check for both "Objects" format and cleaned format
        is_class_header = line.endswith(" Objects") or (
            line.startswith("## ") and re.match(r"^## ([A-Za-z_][A-Za-z0-9_]*)$", line)
        )
        is_function_header = line.startswith("#### ")

        if is_class_header or is_function_header:
            result_lines.append(line)

            # Extract object name and determine type
            object_name = None
            object_type = None

            if line.endswith(" Objects"):
                class_match = re.match(r"^## ([A-Za-z_][A-Za-z0-9_]*) Objects", line)
                object_name = class_match.group(1) if class_match else None
                object_type = "class"
            elif line.startswith("## "):
                class_match = re.match(r"^## ([A-Za-z_][A-Za-z0-9_]*)$", line)
                object_name = class_match.group(1) if class_match else None
                object_type = "class"
            elif line.startswith("#### "):
                func_match = re.match(r"^#### ([a-z_\\][a-z0-9_\\]*)", line)
                object_name = func_match.group(1) if func_match else None
                object_type = "function"

            # Add GitHub source link
            if current_module and object_name and object_type:
                github_link = get_github_source_link(
                    current_module, object_name, object_type
                )
                result_lines.append("")
                result_lines.append(f"[View source on GitHub]({github_link})")
                result_lines.append("")

            continue

        # Just add the line as-is (no internal linking here to avoid double linking)
        result_lines.append(line)

    return "\n".join(result_lines)


def convert_attributes_to_table(
    content: str, all_definitions: Dict[str, Dict[str, str]] = None
) -> str:
    """Convert attribute and argument lists to markdown tables with internal links for custom types."""

    def add_type_links(text: str, in_code_context: bool = False) -> str:
        """Add internal links to custom types in text."""
        if all_definitions is None or in_code_context:
            return text

        # If the text already contains markdown links, don't add more links to avoid double linking
        import re

        if re.search(r"\[([^\]]+)\]\([^)]+\)", text):
            return text

        # Handle complex types like Union[GenerationConfig, GenerationConfigDict]
        # Extract individual type names from complex type expressions
        type_names = re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", text)

        result_text = text
        for type_name in type_names:
            if type_name in all_definitions:
                def_info = all_definitions[type_name]
                if def_info["type"] == "class":  # Only link to classes
                    link_path = f"/{def_info['path']}"
                    # Create a link for this type name
                    pattern = r"\b" + re.escape(type_name) + r"\b"
                    replacement = f"[{type_name}]({link_path})"
                    result_text = re.sub(pattern, replacement, result_text, count=1)

        return result_text

    lines = content.split("\n")
    result_lines = []
    i = 0
    in_code_block = False
    in_frontmatter = False
    frontmatter_start_seen = False

    while i < len(lines):
        line = lines[i]

        # Track if we're inside frontmatter
        if line.strip() == "---":
            if not frontmatter_start_seen:
                # First --- marks start of frontmatter
                frontmatter_start_seen = True
                in_frontmatter = True
                result_lines.append(line)
                i += 1
                continue
            elif in_frontmatter:
                # Second --- marks end of frontmatter
                in_frontmatter = False
                result_lines.append(line)
                i += 1
                continue
            else:
                # Standalone --- (not frontmatter)
                result_lines.append(line)
                i += 1
                continue

        # Skip processing if we're in frontmatter
        if in_frontmatter:
            result_lines.append(line)
            i += 1
            continue

        # Track if we're inside a code block
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result_lines.append(line)
            i += 1
            continue

        # Look for "Attributes:", "Arguments:", or "Returns:" followed by a list
        if (
            line.strip() == "**Attributes**:"
            or line.strip() == "Attributes:"
            or line.strip() == "**Arguments**:"
            or line.strip() == "Arguments:"
            or line.strip() == "**Returns**:"
            or line.strip() == "Returns:"
        ):

            section_type = (
                "Arguments"
                if "Arguments" in line
                else "Returns" if "Returns" in line else "Attributes"
            )
            result_lines.append(line)
            result_lines.append("")  # Add empty line after header
            i += 1

            # Skip any empty lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1

            # Collect all list items first
            table_rows = []
            while i < len(lines):
                current_line = lines[i].strip()

                # Check if this is a list item with different formats:
                # Format 1: - `param` _type_ - description (with type)
                # Format 2: - `param` - description (without type)
                # Format 3: - `Type` - description (Returns section)
                if re.match(r"^-\s+`([^`]+)`\s+_([^_]+)_\s+-\s+(.*)", current_line):
                    # Format 1: with type in underscores
                    match = re.match(
                        r"^-\s+`([^`]+)`\s+_([^_]+)_\s+-\s+(.*)", current_line
                    )
                    if match:
                        param_name = match.group(1).strip()
                        param_type = match.group(2).strip()
                        param_desc = match.group(3).strip()

                        # Add links to the type and description
                        linked_type = add_type_links(param_type, in_code_block)
                        linked_desc = add_type_links(param_desc, in_code_block)

                        table_rows.append(
                            {
                                "name": f"`{param_name}`",
                                "type": f"_{linked_type}_",
                                "description": linked_desc,
                            }
                        )
                elif re.match(r"^-\s+`([^`]+)`\s+-\s+(.*)", current_line):
                    # Format 2/3: Simple format (both Arguments and Returns use this)
                    match = re.match(r"^-\s+`([^`]+)`\s+-\s+(.*)", current_line)
                    if match:
                        item_name = match.group(1).strip()
                        item_desc = match.group(2).strip()

                        # Add links, but avoid linking the same class multiple times in the same line
                        linked_name = add_type_links(item_name, in_code_block)

                        # Check what classes were linked in the name part
                        linked_classes = set()
                        if all_definitions:
                            for class_name in all_definitions:
                                if f"[{class_name}]" in linked_name:
                                    linked_classes.add(class_name)

                        # For the description, create a modified definitions dict that excludes already-linked classes
                        if linked_classes:
                            filtered_definitions = {
                                name: info
                                for name, info in all_definitions.items()
                                if name not in linked_classes
                            }

                            def add_type_links_filtered(
                                text: str, in_code_context: bool = False
                            ) -> str:
                                if filtered_definitions is None or in_code_context:
                                    return text
                                if re.search(r"\[([^\]]+)\]\([^)]+\)", text):
                                    return text
                                type_names = re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", text)
                                result_text = text
                                for type_name in type_names:
                                    if type_name in filtered_definitions:
                                        def_info = filtered_definitions[type_name]
                                        if def_info["type"] == "class":
                                            link_path = f"/{def_info['path']}"
                                            pattern = (
                                                r"\b" + re.escape(type_name) + r"\b"
                                            )
                                            replacement = f"[{type_name}]({link_path})"
                                            result_text = re.sub(
                                                pattern,
                                                replacement,
                                                result_text,
                                                count=1,
                                            )
                                return result_text

                            linked_desc = add_type_links_filtered(
                                item_desc, in_code_block
                            )
                        else:
                            linked_desc = add_type_links(item_desc, in_code_block)

                        # For simple format, check if it looks like a type (for Returns section)
                        if section_type == "Returns" and re.match(
                            r"^[A-Z][a-zA-Z0-9_]*$", item_name
                        ):
                            table_rows.append(
                                {
                                    "name": f"`{linked_name}`",
                                    "type": "",
                                    "description": linked_desc,
                                }
                            )
                        else:
                            table_rows.append(
                                {
                                    "name": f"`{linked_name}`",
                                    "type": "",
                                    "description": linked_desc,
                                }
                            )
                elif current_line == "" or not current_line:
                    # Empty line, add it and continue but don't break
                    pass
                else:
                    # Not a list item, break out of parsing this section
                    break
                i += 1

            # Generate markdown table if we have rows
            if table_rows:
                # Determine if we need a Type column
                has_types = any(row["type"] for row in table_rows)

                if has_types:
                    # Three-column table: Name | Type | Description
                    result_lines.append("| Name | Type | Description |")
                    result_lines.append("|------|------|-------------|")
                    for row in table_rows:
                        result_lines.append(
                            f"| {row['name']} | {row['type']} | {row['description']} |"
                        )
                else:
                    # Two-column table: Name | Description
                    result_lines.append("| Name | Description |")
                    result_lines.append("|------|-------------|")
                    for row in table_rows:
                        result_lines.append(f"| {row['name']} | {row['description']} |")

                result_lines.append("")  # Add empty line after table

            continue
        else:
            # Add internal links to any line that might contain type references
            # Skip if we're in a code block, frontmatter, the line contains backticks, or it's a main title
            if not (
                in_code_block or in_frontmatter or "`" in line or line.startswith("# ")
            ):
                modified_line = add_type_links(line, in_code_block)
                result_lines.append(modified_line)
            else:
                result_lines.append(line)
        i += 1

    return "\n".join(result_lines)


def is_content_meaningful(content: str) -> bool:
    """Check if the content is meaningful enough to create a file."""
    cleaned = clean_content(content)

    # Remove extra whitespace and split into lines
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

    # Filter out just headings, empty content, and GitHub links
    meaningful_lines = [
        line
        for line in lines
        if not line.startswith("#")
        and not line.startswith("**")
        and not line.startswith("[View")
        and len(line) > 10
        and not line.strip() == ""
    ]

    # Must have substantial content - at least one class or function definition
    has_class_or_function = any(
        line.startswith("```python")
        or line.startswith("class ")
        or line.startswith("def ")
        or line.startswith("#### ")  # Function headers
        or line.startswith("## ")  # Class headers
        for line in lines
    )

    # Must have meaningful lines AND actual definitions
    return len(meaningful_lines) >= 3 and has_class_or_function and len(cleaned) >= 100


def parse_toc_structure(content: str) -> Dict[str, List[str]]:
    """Parse the table of contents to extract module structure."""
    lines = content.split("\n")
    modules = {}
    current_module = None

    for line in lines:
        if line.startswith("* [maxim."):
            # Extract module name and clean any markdown links
            match = re.search(r"\[([^]]+)\]", line)
            if match:
                module_name = match.group(1)
                # Clean any nested markdown links from the module name
                module_name = clean_title_text(module_name)
                if module_name not in modules:
                    modules[module_name] = []
                current_module = module_name
        elif line.startswith("  * [") and current_module:
            # Extract submodule name and clean any markdown links
            match = re.search(r"\[([^]]+)\]", line)
            if match:
                submodule_name = match.group(1)
                # Clean any nested markdown links from the submodule name
                submodule_name = clean_title_text(submodule_name)
                modules[current_module].append(submodule_name)

    return modules


def sanitize_filename(module_name: str) -> str:
    """Sanitize module name for use as filename."""
    # Replace dots with forward slashes for directory structure
    # Unescape any backslash-escaped underscores from pydoc-markdown
    sanitized = module_name.replace(".", "/").replace("\\_", "_")
    return sanitized


def create_file_path(base_dir: Path, module_name: str) -> Path:
    """Create the file path for a module."""
    sanitized = sanitize_filename(module_name)
    # Remove the 'maxim/' prefix if it exists since we're already in maxim directory
    if sanitized.startswith("maxim/"):
        sanitized = sanitized[6:]

    file_path = base_dir / f"{sanitized}.mdx"
    return file_path


def create_docs_json_path(module_name: str) -> str:
    """Create the path for docs.json navigation."""
    sanitized = sanitize_filename(module_name)
    # Remove the 'maxim/' prefix if it exists
    if sanitized.startswith("maxim/"):
        sanitized = sanitized[6:]

    return f"sdk/python/references/{sanitized}"


def generate_descriptive_title(
    module_name: str, title_registry: Dict[str, List[str]] = None
) -> str:
    """Generate a basic descriptive title for a module."""
    # Remove 'maxim.' prefix
    clean_name = module_name
    if clean_name.startswith("maxim."):
        clean_name = clean_name[6:]

    # Split into parts
    parts = clean_name.split(".")

    def clean_part(part: str) -> str:
        """Clean a module part by removing backslashes and converting underscores to spaces."""
        return part.replace("\\_", "_").replace("_", " ")

    if len(parts) == 1:
        # Top-level module, just capitalize and clean underscores
        title = clean_part(parts[0]).title()
        return clean_title_text(title)

    # For nested modules, create a descriptive title
    last_part = parts[-1]
    parent_parts = parts[:-1]

    # Clean underscores from all parts
    clean_last_part = clean_part(last_part)

    if last_part == "__init__":
        # For __init__ modules, use the parent name
        clean_parent_parts = [clean_part(part) for part in parent_parts]
        title = " ".join(part.title() for part in clean_parent_parts)
        return clean_title_text(title)
    else:
        # For all other modules, convert underscores to camel case
        raw_last_part = parts[-1].replace("\\_", "_")
        camel_case_part = "".join(
            word.capitalize() for word in raw_last_part.split("_")
        )
        return clean_title_text(camel_case_part)


def generate_module_description(module_name: str) -> str:
    """Generate a descriptive description for a module."""
    # Remove 'maxim.' prefix
    clean_name = module_name
    if clean_name.startswith("maxim."):
        clean_name = clean_name[6:]

    # Split into parts
    parts = clean_name.split(".")

    # Module-specific descriptions
    module_descriptions = {
        "maxim": "Core Maxim Python SDK functionality and main entry point.",
        "logger": "Logging and instrumentation utilities for tracking AI model interactions.",
        "decorators": "Decorators for automatic logging and instrumentation of functions and methods.",
        "models": "Data models and type definitions used throughout the Maxim SDK.",
        "apis": "API client utilities for interacting with Maxim services.",
        "cache": "Caching mechanisms and utilities for optimizing performance.",
        "dataset": "Dataset management and manipulation utilities.",
        "evaluators": "Evaluation tools and utilities for assessing model performance.",
        "runnable": "Runnable interfaces and execution utilities.",
        "test_runs": "Test execution and management utilities.",
        "utils": "Common utility functions and helpers.",
        "types": "Type definitions and data structures.",
        "anthropic": "Anthropic AI model integration and logging utilities.",
        "openai": "OpenAI model integration and logging utilities.",
        "gemini": "Google Gemini model integration and logging utilities.",
        "langchain": "LangChain framework integration utilities.",
        "litellm": "LiteLLM integration for multi-provider AI model access.",
        "bedrock": "AWS Bedrock integration utilities.",
        "mistral": "Mistral AI model integration utilities.",
        "portkey": "Portkey integration utilities.",
        "crewai": "CrewAI framework integration utilities.",
        "livekit": "LiveKit real-time communication integration utilities.",
    }

    if len(parts) == 1:
        # Top-level module
        base_name = parts[0].replace("\\_", "_")
        return module_descriptions.get(
            base_name, f"{parts[0].title()} module utilities and functionality."
        )

    # For nested modules, create contextual descriptions
    parent_part = parts[-2].replace("\\_", "_") if len(parts) > 1 else ""
    last_part = parts[-1].replace("\\_", "_")

    # Special handling for common patterns
    if last_part == "client":
        provider = parent_part.title() if parent_part else "Service"
        return f"{provider} client implementation for API interactions and model integration."
    elif last_part == "utils":
        provider = parent_part.title() if parent_part else "Module"
        return f"Utility functions and helpers for {provider} integration."
    elif last_part == "tracer":
        provider = parent_part.title() if parent_part else "Service"
        return f"Tracing and instrumentation utilities for {provider} integration."
    elif parent_part in module_descriptions:
        base_desc = module_descriptions[parent_part]
        return f"{last_part.title()} utilities for {base_desc.lower()}"
    else:
        # Generic description
        clean_last = last_part.replace("_", " ").title()
        clean_parent = (
            parent_part.replace("_", " ").title() if parent_part else "Maxim SDK"
        )
        return f"{clean_last} functionality for {clean_parent} integration."


def resolve_title_collisions(module_titles: Dict[str, str]) -> Dict[str, str]:
    """Resolve title collisions by adding parent context to duplicate titles."""
    # Group modules by their generated titles
    title_groups = {}
    for module_name, title in module_titles.items():
        if title not in title_groups:
            title_groups[title] = []
        title_groups[title].append(module_name)

    # Resolve collisions
    resolved_titles = {}
    for title, modules in title_groups.items():
        if len(modules) == 1:
            # No collision, keep original title
            resolved_titles[modules[0]] = title
        else:
            # Collision detected, add parent context to all conflicting modules
            for module_name in modules:
                clean_name = module_name
                if clean_name.startswith("maxim."):
                    clean_name = clean_name[6:]

                parts = clean_name.split(".")

                def clean_part(part: str) -> str:
                    return part.replace("\\_", "_").replace("_", " ")

                if len(parts) >= 2:
                    # Add one level of parent context using dot notation
                    parent_part = parts[-2].replace(
                        "\\_", "_"
                    )  # Keep original case for parent
                    last_part = parts[-1].replace("\\_", "_")
                    # Convert underscores to camel case for the entity
                    camel_case_part = "".join(
                        word.capitalize() for word in last_part.split("_")
                    )
                    title = f"{parent_part}.{camel_case_part}"
                    resolved_titles[module_name] = clean_title_text(title)
                elif len(parts) == 1:
                    # Fallback to original title if no parent available
                    resolved_titles[module_name] = clean_title_text(title)

    return resolved_titles


def extract_module_content(content: str, module_name: str) -> str:
    """Extract content for a specific module."""
    lines = content.split("\n")
    start_idx = None
    end_idx = len(lines)

    # Find the start of this module's content - look for # heading
    for i, line in enumerate(lines):
        if line.strip() == f"# {module_name}":
            start_idx = i
            break

    if start_idx is None:
        return ""

    # Find the end (next # heading at the same level)
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith("# ") and not lines[i].startswith("## "):
            end_idx = i
            break

    module_content = "\n".join(lines[start_idx:end_idx])

    # Generate descriptive title for frontmatter
    descriptive_title = generate_descriptive_title(module_name)

    # Remove the original H1 title and replace with MDX frontmatter
    module_content = module_content.replace(f"# {module_name}", "", 1)

    # Add module-level GitHub source link
    module_github_link = get_github_source_link(module_name)

    # Create MDX frontmatter and content
    frontmatter = f"""---
title: {descriptive_title}
---

[View module source on GitHub]({module_github_link})

"""

    # Combine frontmatter with cleaned content
    module_content = frontmatter + module_content.lstrip()

    return module_content


def create_hierarchical_docs_structure(created_files: List[str]) -> Dict:
    """Create a hierarchical structure for docs.json based on module paths."""
    # Group modules by their top-level directory
    groups = {}

    for module in created_files:
        # Get the sanitized path
        path = sanitize_filename(module)
        if path.startswith("maxim/"):
            path = path[6:]

        # Split into parts
        parts = path.split("/")
        if len(parts) == 1:
            # Top-level module
            top_level = "Core"
        else:
            # Use the first directory as the group name
            top_level = parts[0].title()

        if top_level not in groups:
            groups[top_level] = []

        groups[top_level].append(create_docs_json_path(module))

    # Convert to the format expected by docs.json
    pages = []
    for group_name, group_pages in sorted(groups.items()):
        if group_name == "Core":
            # Add core modules directly to the main references
            pages.extend(group_pages)
        else:
            # Create a subgroup for this directory
            pages.append({"group": group_name.title(), "pages": sorted(group_pages)})

    return {
        "navigation": {
            "tabs": [
                {
                    "tab": "SDK",
                    "groups": [
                        {
                            "group": "Python",
                            "pages": [{"group": "References", "pages": pages}],
                        }
                    ],
                }
            ]
        }
    }


def create_custom_title_script(output_dir: str) -> None:
    """Create a custom JavaScript file to update page titles with '| Maxim Python SDK'."""
    script_content = """// Custom script to update page titles with Maxim Python SDK suffix
(function() {
    function isPythonSDKPage() {
        // Check if current page is a Python SDK reference page
        const currentPath = window.location.pathname;
        
        // Check if the path matches Python SDK reference patterns
        return currentPath.includes('/sdk/python/references/') || 
               currentPath.includes('/sdk/python/reference/') ||
               currentPath.includes('/python/references/') ||
               currentPath.includes('/python/reference/') ||
               // Also check for specific patterns in the URL that indicate Python SDK pages
               (currentPath.includes('/python') && (
                   currentPath.includes('/maxim') || 
                   currentPath.includes('/decorators') || 
                   currentPath.includes('/logger') || 
                   currentPath.includes('/models') ||
                   currentPath.includes('/apis') ||
                   currentPath.includes('/cache') ||
                   currentPath.includes('/dataset') ||
                   currentPath.includes('/evaluators') ||
                   currentPath.includes('/runnable') ||
                   currentPath.includes('/test_runs') ||
                   currentPath.includes('/utils')
               ));
    }
    
    function updatePageTitle() {
        // Only update title if this is a Python SDK page
        if (!isPythonSDKPage()) {
            return;
        }
        
        // Get the current page title from the frontmatter or h1
        const titleElement = document.querySelector('h1') || document.querySelector('[data-title]');
        let pageTitle = '';
        
        // Try to get title from various sources
        if (titleElement) {
            pageTitle = titleElement.textContent || titleElement.innerText || '';
        }
        
        // Fallback to document title if no h1 found
        if (!pageTitle) {
            pageTitle = document.title;
        }
        
        // Clean up the title (remove extra whitespace)
        pageTitle = pageTitle.trim();
        
        // Only update if we have a meaningful title and it doesn't already include our suffix
        if (pageTitle && !pageTitle.includes('| Maxim Python SDK')) {
            // Remove any existing suffixes that might be there
            pageTitle = pageTitle.replace(/\\s*\\|.*$/, '');
            
            // Add our custom suffix
            document.title = pageTitle + ' | Maxim Python SDK';
        }
    }
    
    // Update title when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updatePageTitle);
    } else {
        updatePageTitle();
    }
    
    // Also update when navigating (for SPAs)
    if (window.history && window.history.pushState) {
        const originalPushState = window.history.pushState;
        window.history.pushState = function() {
            originalPushState.apply(window.history, arguments);
            setTimeout(updatePageTitle, 100); // Small delay to ensure DOM is updated
        };
        
        const originalReplaceState = window.history.replaceState;
        window.history.replaceState = function() {
            originalReplaceState.apply(window.history, arguments);
            setTimeout(updatePageTitle, 100);
        };
        
        window.addEventListener('popstate', function() {
            setTimeout(updatePageTitle, 100);
        });
    }
    
    // Watch for title changes using MutationObserver
    if (typeof MutationObserver !== 'undefined') {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList' || mutation.type === 'characterData') {
                    // Check if title-related elements changed
                    const titleChanged = Array.from(mutation.addedNodes).some(node => 
                        node.nodeType === Node.ELEMENT_NODE && 
                        (node.tagName === 'H1' || node.querySelector && node.querySelector('h1'))
                    );
                    
                    if (titleChanged || mutation.target.tagName === 'TITLE') {
                        setTimeout(updatePageTitle, 50);
                    }
                }
            });
        });
        
        observer.observe(document, {
            childList: true,
            subtree: true,
            characterData: true
        });
        
        // Also observe the title element specifically
        const titleElement = document.querySelector('title');
        if (titleElement) {
            observer.observe(titleElement, {
                childList: true,
                characterData: true
            });
        }
    }
})();"""

    # Write the JavaScript file
    script_path = os.path.join(output_dir, "title-updater.js")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    print(f"Created custom title script: {script_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python split_docs.py <input_file> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Read the input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all class and function definitions across all modules
    all_definitions = find_class_and_function_definitions(content)
    print(
        f"Found {len(all_definitions)} class/function definitions for internal linking"
    )

    # Parse the table of contents to get module structure
    modules = parse_toc_structure(content)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create custom JavaScript file for title updates
    create_custom_title_script(output_dir)

    # First pass: generate initial titles and check for collisions
    module_titles = {}
    meaningful_modules = []

    for module_name in modules:
        # Skip the top-level 'maxim' module
        if module_name == "maxim":
            continue

        # Extract module content
        module_content = extract_module_content(content, module_name)

        # Check if content is meaningful
        if not is_content_meaningful(module_content):
            print(f"Skipping {module_name} (empty/minimal content)")
            continue

        meaningful_modules.append(module_name)
        module_titles[module_name] = generate_descriptive_title(module_name)

    # Resolve title collisions
    resolved_titles = resolve_title_collisions(module_titles)

    # Second pass: create files with resolved titles
    created_files = []
    for module_name in meaningful_modules:
        # Get the resolved title
        resolved_title = resolved_titles[module_name]

        # Extract module content with custom title
        lines = content.split("\n")
        start_idx = None
        end_idx = len(lines)

        # Find the start of this module's content - look for # heading
        for i, line in enumerate(lines):
            if line.strip() == f"# {module_name}":
                start_idx = i
                break

        if start_idx is None:
            continue

        # Find the end (next # heading at the same level)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith("# ") and not lines[i].startswith("## "):
                end_idx = i
                break

        module_content = "\n".join(lines[start_idx:end_idx])

        # Remove the original H1 title
        module_content = module_content.replace(f"# {module_name}", "", 1)

        # Add module-level GitHub source link
        module_github_link = get_github_source_link(module_name)

        # Generate description for this module
        module_description = generate_module_description(module_name)

        # Create MDX frontmatter with resolved title and description
        frontmatter = f"""---
title: {resolved_title}
description: {module_description}
---

[View module source on GitHub]({module_github_link})

"""

        # Combine frontmatter with cleaned content
        module_content = frontmatter + module_content.lstrip()

        # Add GitHub source links
        enhanced_content = add_github_source_links(module_content, all_definitions)

        # Clean the content (this should be done AFTER adding links to ensure anchor tags are removed)
        cleaned_content = clean_content(enhanced_content, all_definitions)

        # Create file path
        file_path = create_file_path(Path(output_dir), module_name)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)

        print(f"Created: {file_path} (title: {resolved_title})")
        created_files.append(module_name)

    # Create hierarchical docs.json for Mintlify
    docs_json = create_hierarchical_docs_structure(created_files)

    docs_json_path = os.path.join(output_dir, "docs.json")
    with open(docs_json_path, "w", encoding="utf-8") as f:
        json.dump(docs_json, f, indent=2)

    print(f"Created docs.json with {len(created_files)} module references")

    # Clean up the temporary file
    if os.path.exists(input_file):
        os.remove(input_file)
        print(f"Cleaned up temporary file: {input_file}")

    print(
        "Documentation generated in docs/ directory with internal links and GitHub source links"
    )


if __name__ == "__main__":
    main()
