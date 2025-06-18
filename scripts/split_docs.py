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

    while i < len(lines):
        line = lines[i]

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
            i += 1

            # Skip any empty lines
            while i < len(lines) and lines[i].strip() == "":
                result_lines.append(lines[i])
                i += 1

            # Look for the start of the list and add internal links
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

                        result_lines.append(
                            f"- `{param_name}` _{linked_type}_ - {linked_desc}"
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
                            # Temporarily modify the function's closure to use filtered definitions
                            original_definitions = all_definitions

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

                        result_lines.append(f"- `{linked_name}` - {linked_desc}")
                elif current_line == "" or not current_line:
                    # Empty line, add it and continue
                    result_lines.append(lines[i])
                else:
                    # Not a list item, break out of parsing this section
                    break
                i += 1
            continue
        else:
            # Add internal links to any line that might contain type references
            # Skip if we're in a code block or the line contains backticks
            if not (in_code_block or "`" in line):
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

    # Filter out just headings and empty content
    meaningful_lines = [
        line
        for line in lines
        if not line.startswith("#") and not line.startswith("**") and len(line) > 10
    ]

    # Must have at least 3 meaningful lines or 50+ characters of content
    return len(meaningful_lines) >= 2 or len(cleaned) >= 50


def parse_toc_structure(content: str) -> Dict[str, List[str]]:
    """Parse the table of contents to extract module structure."""
    lines = content.split("\n")
    modules = {}
    current_module = None

    for line in lines:
        if line.startswith("* [maxim."):
            # Extract module name
            match = re.search(r"\[([^]]+)\]", line)
            if match:
                module_name = match.group(1)
                if module_name not in modules:
                    modules[module_name] = []
                current_module = module_name
        elif line.startswith("  * [") and current_module:
            # Extract submodule name
            match = re.search(r"\[([^]]+)\]", line)
            if match:
                submodule_name = match.group(1)
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

    # Process each module
    created_files = []
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

        print(f"Created: {file_path}")
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
