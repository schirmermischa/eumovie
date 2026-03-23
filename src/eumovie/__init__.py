from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("eumovie")
except PackageNotFoundError:
    __version__ = "unknown"
