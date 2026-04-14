"""
FileRepository.py
Shared MAT-file path resolution, discovery, and thin IO helpers for the project.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import scipy.io

import ControllerConfiguration as Config


@dataclass(frozen=True)
class FileSelection:
    data_type: str  # "secondary" or "collected"
    participant: int
    movement: int


class DataRepository:
    """Handles path resolution, dataset discovery, and thin MAT-file IO helpers."""

    def __init__(self, base_path: Optional[str] = None) -> None:
        self.base_path = os.path.normpath(base_path or getattr(Config, "BASE_DATA_PATH", "./biosignal_data"))

    def raw_root(self, data_type: str) -> str:
        return os.path.join(self.base_path, data_type, "raw")

    def edited_root(self, data_type: str) -> str:
        return os.path.join(self.base_path, data_type, "edited")

    def raw_file_path(self, selection: FileSelection) -> str:
        if selection.data_type == "secondary":
            return os.path.join(
                self.raw_root("secondary"),
                f"Soggetto{selection.participant}",
                f"Movimento{selection.movement}.mat",
            )
        if selection.data_type == "collected":
            return os.path.join(
                self.raw_root("collected"),
                f"P{selection.participant}M{selection.movement}.mat",
            )
        raise ValueError(f"Unknown data type: {selection.data_type}")

    def output_file_path(self, selection: FileSelection, create_dirs: bool = True) -> str:
        raw_path = self.raw_file_path(selection)
        input_basename = os.path.splitext(os.path.basename(raw_path))[0]
        out_name = f"{input_basename}_labelled.mat"

        if selection.data_type == "secondary":
            out_dir = os.path.join(self.edited_root("secondary"), f"Soggetto{selection.participant}")
        elif selection.data_type == "collected":
            out_dir = self.edited_root("collected")
        else:
            raise ValueError(f"Unknown data type: {selection.data_type}")

        if create_dirs:
            os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, out_name)

    def preferred_input_path(self, selection: FileSelection) -> str:
        """Load edited-labelled file first if it exists, otherwise load raw source."""
        edited_path = self.output_file_path(selection, create_dirs=False)
        if os.path.exists(edited_path):
            return edited_path
        return self.raw_file_path(selection)

    def load_mat(self, file_path: str) -> Dict[str, object]:
        return scipy.io.loadmat(file_path)

    def load_preferred_mat(self, selection: FileSelection) -> Tuple[Dict[str, object], str]:
        file_path = self.preferred_input_path(selection)
        return self.load_mat(file_path), file_path

    def save_mat(self, file_path: str, payload: Dict[str, object], create_dirs: bool = True) -> None:
        if create_dirs:
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        scipy.io.savemat(file_path, payload)

    def is_readable_mat(self, file_path: str, required_keys: Sequence[str] = ("EMGDATA",)) -> bool:
        try:
            mat = scipy.io.loadmat(file_path)
        except Exception:
            return False

        return all(key in mat for key in required_keys)

    def discover_participants(self, data_type: str) -> List[int]:
        raw_root = self.raw_root(data_type)
        participants: set[int] = set()

        if data_type == "secondary":
            if os.path.isdir(raw_root):
                for name in os.listdir(raw_root):
                    match = re.match(r"Soggetto(\d+)$", name)
                    if match:
                        participants.add(int(match.group(1)))
        elif data_type == "collected":
            if os.path.isdir(raw_root):
                for name in os.listdir(raw_root):
                    match = re.match(r"P(\d+)M(\d+)\.mat$", name)
                    if match:
                        participants.add(int(match.group(1)))
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if participants:
            return sorted(participants)

        # Fallback defaults when folders are empty or not yet created.
        return list(range(1, 9)) if data_type == "secondary" else list(range(1, 4))

    def discover_movements(self, data_type: str, participant: int) -> List[int]:
        raw_root = self.raw_root(data_type)
        movements: set[int] = set()

        if data_type == "secondary":
            subject_dir = os.path.join(raw_root, f"Soggetto{participant}")
            if os.path.isdir(subject_dir):
                for name in os.listdir(subject_dir):
                    match = re.match(r"Movimento(\d+)\.mat$", name)
                    if match:
                        movements.add(int(match.group(1)))
        elif data_type == "collected":
            if os.path.isdir(raw_root):
                pattern = re.compile(rf"P{participant}M(\d+)\.mat$")
                for name in os.listdir(raw_root):
                    match = pattern.match(name)
                    if match:
                        movements.add(int(match.group(1)))
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        return sorted(movements) if movements else list(range(1, 10))

    def get_all_participant_files(self, participant_id: int, data_type: str) -> List[str]:
        """
        Returns all available movement files for a participant, preferring edited files
        when present and falling back to raw files.
        """
        file_paths: List[str] = []
        for movement in self.discover_movements(data_type, participant_id):
            selection = FileSelection(
                data_type=data_type,
                participant=participant_id,
                movement=movement,
            )
            path = self.preferred_input_path(selection)
            if os.path.exists(path):
                file_paths.append(path)
        return sorted(set(file_paths))
