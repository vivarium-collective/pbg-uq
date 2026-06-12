import json
from dataclasses import fields
from pathlib import Path

import numpy as np


class DataclassIO:
    @classmethod
    def save(cls, instance, path: str | Path):
        """Save any dataclass with numpy arrays to disk.
        :param instance: @dataclass-decorated instance
        :param path: destination to which a dir of artifacts will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {}
        primitives = {}
        for f in fields(instance):
            value = getattr(instance, f.name)
            if isinstance(value, np.ndarray):
                arrays[f.name] = value
            elif value is None or isinstance(value, (str, int, float, bool, list, dict)):
                primitives[f.name] = value
            else:
                # Fallback: convert to string repr (or handle nested dataclasses)
                primitives[f.name] = str(value)
        # Save arrays as .npy files
        for name, arr in arrays.items():
            np.save(path / f"{name}.npy", arr)
        # Save primitives as JSON
        with open(path / "metadata.json", "w") as f:
            json.dump(primitives, f, indent=2)
        # Save class name for reconstruction
        with open(path / "class.txt", "w") as f:
            f.write(f"{instance.__class__.__module__}.{instance.__class__.__name__}")

    @classmethod
    def load(cls, path: str | Path, _class):
        """Load a dataclass from disk."""
        path = Path(path)
        # Load primitives
        with open(path / "metadata.json") as f:
            data = json.load(f)
        # Load arrays
        for npy_file in path.glob("*.npy"):
            name = npy_file.stem
            data[name] = np.load(npy_file)
        return _class(**data)
