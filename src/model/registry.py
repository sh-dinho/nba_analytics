"""
Model Registry for NBA Analytics v3.
Handles versioning, saving, and loading models + metadata.
"""

from pathlib import Path
import json
import joblib
from loguru import logger


class ModelRegistry:
    def __init__(self, registry_dir="models/registry"):
        """
        Initializes the ModelRegistry with a specified directory.
        If the directory doesn't exist, it will be created.

        Args:
            registry_dir (str): Path to the directory where models and metadata will be stored.
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def next_version(self) -> int:
        """
        Determine the next available version for the model based on existing versions.

        Returns:
            int: The next version number.
        """
        versions = []
        for p in self.registry_dir.glob("model_*.joblib"):
            try:
                versions.append(
                    int(p.stem.split("_")[1])
                )  # Extract version number from filename
            except ValueError:
                continue
        return max(versions, default=0) + 1

    def save(self, model, metadata: dict, version: int):
        """
        Saves the model and its associated metadata to the registry.

        Args:
            model: The trained machine learning model to save.
            metadata (dict): A dictionary containing metadata about the model (e.g., features, target, model type).
            version (int): The version number for the model.
        """
        model_path = self.registry_dir / f"model_{version}.joblib"
        meta_path = self.registry_dir / f"model_{version}.meta.json"

        # Save atomically to avoid partial writes
        tmp_model_path = model_path.with_suffix(".tmp")
        joblib.dump(model, tmp_model_path)  # Save model to temp file
        tmp_model_path.rename(model_path)  # Rename temp file to final file path

        tmp_meta_path = meta_path.with_suffix(".tmp")
        with open(tmp_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)  # Save metadata to temp file
        tmp_meta_path.rename(meta_path)  # Rename temp file to final file path

        logger.info(f"Saved model → {model_path}")
        logger.info(f"Saved metadata → {meta_path}")

    def load_latest(self):
        """
        Load the latest model based on its version.

        Returns:
            model: The most recently saved model.
            metadata: The metadata associated with the model.
            version: The version number of the model.
        """
        models = list(self.registry_dir.glob("model_*.joblib"))
        if not models:
            return None, None, None

        # Sort models by numeric version
        def version_num(p: Path):
            try:
                return int(p.stem.split("_")[1])  # Extract version number from filename
            except ValueError:
                return -1  # Invalid files should go first

        models = sorted(models, key=version_num)
        latest = models[-1]
        version = int(latest.stem.split("_")[1])

        # Load the metadata for the latest model
        meta_path = self.registry_dir / f"model_{version}.meta.json"
        if not meta_path.exists():
            logger.warning(f"Metadata file missing for version {version}")
            metadata = None
        else:
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        model = joblib.load(latest)  # Load the model
        return model, metadata, version
