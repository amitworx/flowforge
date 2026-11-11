def generate_cloned_workflow_name(source_name: str) -> str:
    suffix = " (copy)"
    max_len = 128
    if len(source_name) + len(suffix) <= max_len:
        return f"{source_name}{suffix}"
    allowed_source_len = max_len - len(suffix)
    return f"{source_name[:allowed_source_len]}{suffix}"
