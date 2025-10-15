# Agent rules

| Category               | Rule                                                      |
| ---------------------- | --------------------------------------------------------- |
| Naming                 | snake_case everywhere, SCREAMING_SNAKE_CASE for constants |
| Documentation          | Docstring (NumPy or Google style) above every function    |
| Structure              | Clear file hierarchy and modular separation               |
| Typing                 | Explicit type annotations for public functions and variables |
| Testing                | Mandatory unit tests per exposed function                 |
| Communication          | Use defined interfaces, avoid tight coupling              |
| Error handling         | Descriptive errors with `[AGENT_NAME]` prefix             |
| Security & Performance | Sanitize, optimize, and version properly                  |

This document codifies the agent development rules used in this repository. Follow these guidelines for all code contributions, scripts, and automated agents.

## Rules and rationale
- Ignore
  - Ignore all venv, .ipynb_checkpoint, .git or __pycache__ folders and their content. They do not contain any relevant content
- Context
  - Almost any code context required will be found in context/ folder. There might be .sql database structures...
- Naming
  - Use snake_case for all variable names, function names, and module-level identifiers.
  - Use SCREAMING_SNAKE_CASE for constants (module-level immutables such as CONFIG_PATH, MAX_RETRIES).
  - Rationale: consistent naming reduces cognitive load and simplifies search/analysis.

- Documentation
  - Every new function (public or private) must include a preceding Python docstring (NumPy or Google style) describing:
    - purpose, parameters (name, type, description), return value, and possible exceptions.
  - Example NumPy-style docstring (for reference):

```python
def short_description_of_function(param_name: str) -> dict:
    """Short description of function.

    Parameters
    ----------
    param_name : str
        Description of the parameter.

    Returns
    -------
    dict
        Description of the return value.

    Raises
    ------
    ValueError
        [AGENT_NAME] descriptive error message
    """
    if not isinstance(param_name, str):
        raise ValueError('[AGENT_NAME] param_name must be a string')
    # implementation
    return {}
```

- Structure
  - Organize code with clear module boundaries. Each module should have a single responsibility.
  - Prefer smaller modules with well-defined interfaces exported.
  - Use a predictable directory layout (for example: `src/`, `utils/`, `tests/`).

- Typing
  - All public function signatures and module-level declarations must include explicit type annotations.
  - Use Python typing (PEP 484) with concrete types (e.g., `int`, `str`, `List[str]`, `Dict[str, float]`).
  - Avoid un-annotated `Any` for public APIs. Internal helper functions may use `Any` sparingly but should include a justification in comments.
  - Example:

```python
from typing import Dict

MAX_RETRIES: int = 3

def compute_feature_from_pgn(pgn_text: str) -> Dict[str, float]:
    """Compute feature vector from PGN text.

    Parameters
    ----------
    pgn_text : str
        PGN game text.

    Returns
    -------
    Dict[str, float]
        Extracted numeric features.
    """
    # implementation
    return {}
```


- Testing
  - Provide unit tests for every exported function and any non-trivial internal function.
  - Tests should be placed under a `tests/` directory mirroring the `src/` layout.
  - Test names and fixtures should use snake_case as well.

- Communication / interfaces
  - Define and document public interfaces for modules. Avoid tight coupling; prefer dependency injection where appropriate.

- Error handling
  - Errors thrown by agent code must include an `[AGENT_NAME]` prefix to facilitate log filtering.
  - Provide detailed, actionable messages and avoid leaking sensitive information.

- Security & Performance
  - Sanitize and validate external inputs.
  - Avoid naive string concatenation for commands or SQL; prefer parameterized approaches.
  - Profile and optimize known hot paths; include caching or rate limiting where appropriate.

## Enforcement and examples

- Linting
  - A repository lint configuration should enforce naming conventions and basic style.

- Pre-commit
  - Use pre-commit hooks to run linters and unit tests for staged changes.

- Example: compliant function (Python)

```python
def compute_feature_from_pgn(pgn_text: str) -> dict:
  """Compute feature vector from PGN text.

  Parameters
  ----------
  pgn_text : str
    PGN game text.

  Returns
  -------
  dict
    Extracted numeric features.

  Raises
  ------
  TypeError
    [AGENT_NAME] invalid PGN input
  """
  if not isinstance(pgn_text, str):
    raise TypeError('[AGENT_NAME] pgn_text must be a string')
  # ...implementation...
  return {}
```

## Notes

- The rules above apply to all languages and scripts used by the project. When language-specific conventions conflict with these rules, prefer consistency with the repository-level guidelines unless there is a compelling reason not to.

If you want, I can also:
- Add a lint configuration (ESLint / style rules) that enforces snake_case and JSDoc requirements.
- Add a pre-commit configuration to run tests and linters.