# Adapted from flowforms.
"""Scene description dataclasses for cartan-viz renders."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Surface:
    """Colormapped surface layer showing the B scalar field."""

    enabled: bool = True
    field: str = "B"
    cmap: str | None = None  # None uses brand.field_cmap("sequential")


@dataclass
class Glyphs:
    """Arrow glyph layer showing the E vector field at cell centers."""

    enabled: bool = True
    field: str = "E"
    factor: float = 0.05


@dataclass
class Scene:
    """Complete scene description for a single cartan frame render."""

    surface: Surface = field(default_factory=Surface)
    glyphs: Glyphs = field(default_factory=Glyphs)
    background: str | None = None
    orbit: bool = False  # 2D hero is top-down; orbit off by default

    @classmethod
    def default(cls) -> "Scene":
        """Return a Scene with all default settings."""
        return cls()
