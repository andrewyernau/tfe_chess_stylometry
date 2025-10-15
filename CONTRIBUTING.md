# Guía de Contribución

Guía para contribuir al proyecto Chess Stylometry.

---

## 🎯 Convenciones de Código

### Naming Conventions

```python
# Variables y funciones: snake_case
def parse_pgn_file(file_path: str) -> List[Game]:
    game_count = 0
    parsed_games = []
    
# Constantes: SCREAMING_SNAKE_CASE
MAX_GAMES_PER_PLAYER = 100
DEFAULT_BATCH_SIZE = 32

# Clases: PascalCase
class ChessEncoder:
    pass
```

### Documentación

Todas las funciones públicas deben tener docstrings:

```python
def generate_heatmap(board: chess.Board, player: str) -> np.ndarray:
    """
    Genera un mapa de calor de movimientos para un jugador.
    
    Parameters
    ----------
    board : chess.Board
        Tablero de ajedrez actual
    player : str
        'white' o 'black'
    
    Returns
    -------
    np.ndarray
        Array 8x8 con frecuencias normalizadas
    
    Examples
    --------
    >>> board = chess.Board()
    >>> heatmap = generate_heatmap(board, 'white')
    >>> heatmap.shape
    (8, 8)
    """
    pass
```

### Type Hints

Usar type annotations en todas las funciones:

```python
from typing import List, Dict, Optional, Tuple

def process_games(
    pgn_path: str,
    max_games: Optional[int] = None
) -> Tuple[List[Game], Dict[str, int]]:
    """Procesa partidas PGN."""
    pass
```

---

## 🧪 Testing

### Estructura de Tests

```
labs/tests/
├── test_parsers.py
├── test_encoders.py
├── test_models.py
└── test_utils.py
```

### Ejemplo de Test

```python
import pytest
from labs.src.parsers import parse_pgn_file

def test_parse_pgn_file():
    """Test parsing de archivo PGN válido."""
    games = parse_pgn_file("labs/dataset/chessgame0001.pgn")
    assert len(games) > 0
    assert games[0].headers["Event"] is not None

def test_parse_invalid_pgn():
    """Test manejo de PGN inválido."""
    with pytest.raises(ValueError):
        parse_pgn_file("invalid.pgn")
```

---

## 📝 Commits

### Formato de Commits

Seguir [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Implementar generador de mapas de calor
fix: Corregir parsing de movimientos especiales
docs: Actualizar README con ejemplos
test: Añadir tests para encoder
refactor: Simplificar pipeline de datos
style: Formatear código con black
```

### Ejemplos

```bash
git commit -m "feat: Añadir encoder de trayectorias temporales"
git commit -m "fix: Resolver error en split de dataset"
git commit -m "docs: Actualizar documentación de arquitectura"
```

---

## 🌿 Workflow de Git

### Crear Feature Branch

```bash
# Actualizar main
git checkout main
git pull origin main

# Crear rama
git checkout -b feature/nombre-feature

# Trabajar...
git add .
git commit -m "feat: descripción"

# Push
git push origin feature/nombre-feature
```

### Tipos de Ramas

- `feature/`: Nueva funcionalidad
- `fix/`: Corrección de bugs
- `docs/`: Documentación
- `refactor/`: Refactorización
- `test/`: Tests

---

## 🔧 Herramientas

### Formateo de Código

```bash
# Black (code formatter)
black labs/src/

# Flake8 (linter)
flake8 labs/src/
```

### Tests

```bash
# Ejecutar todos los tests
pytest labs/tests/

# Con coverage
pytest --cov=labs/src labs/tests/

# Tests específicos
pytest labs/tests/test_parsers.py -v
```

---

## 📚 Recursos

- [PEP 8](https://peps.python.org/pep-0008/): Guía de estilo Python
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/): Formato de docstrings
- [Type Hints](https://docs.python.org/3/library/typing.html): Type annotations

---

Para más detalles, ver [docs/agents/AGENTS.md](docs/agents/AGENTS.md).
