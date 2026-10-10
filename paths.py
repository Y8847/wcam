"""Куда складывать то, что порождают скрипты.

Всё сгенерированное идёт в каталог `out/` рядом с этими файлами. Имя,
в котором уже есть каталог, возвращается как есть -- значит можно
указать любой путь явно и он будет соблюдён.
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'out')


def resolve(name, create=True):
    """Полный путь для порождаемого файла с именем `name`."""
    if os.path.dirname(name):
        return name
    if create:
        os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)
