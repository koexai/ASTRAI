"""Central path and source-provenance helpers.

Package resources are resolved from the installed distribution. User inputs
and outputs remain relative to the caller's current working directory unless
an absolute path is supplied.
"""

from importlib import resources
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def source_checkout_root():
    """Return the checkout containing this package, or ``None`` when installed.

    The search follows the package location, never the process working
    directory, so a run launched from an unrelated Git repository cannot be
    attributed to that repository accidentally.
    """
    for candidate in PACKAGE_ROOT.parents:
        pyproject = candidate / "pyproject.toml"
        package = candidate / "src" / "astrai"
        if pyproject.is_file() and package.resolve() == PACKAGE_ROOT:
            return candidate
    return None


def source_snapshot_root():
    """Return the complete checkout when available, else installed sources."""
    return source_checkout_root() or PACKAGE_ROOT


def default_config_path(filename):
    """Return an installed ASTRAI configuration resource as a filesystem path."""
    resource = resources.files("astrai.configs").joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged configuration not found: {filename}")
    return Path(resource)


def resolve_config_path(path, default_filename):
    """Resolve an explicit config or select an installed default resource."""
    if path is None:
        return default_config_path(default_filename)
    return Path(path).expanduser().resolve()


def resolve_user_path(path):
    """Resolve a user-supplied input or output against the current directory."""
    return Path(path).expanduser().resolve()
