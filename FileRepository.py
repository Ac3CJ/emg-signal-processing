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

    @classmethod
    def from_standard_path(cls, folder_path: Optional[str]) -> Optional["DataRepository"]:
        """Build a repository for biosignal_data/.../raw or biosignal_data/.../edited paths."""
        if not folder_path:
            return None

        norm_path = os.path.normpath(folder_path)
        if os.path.basename(norm_path) not in {"raw", "edited"}:
            return None

        data_type = os.path.basename(os.path.dirname(norm_path))
        if data_type not in {"secondary", "collected"}:
            return None

        base_path = os.path.dirname(os.path.dirname(norm_path))
        return cls(base_path=base_path)

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

    def secondary_subject_root(self, participant: int) -> str:
        return os.path.join(self.raw_root("secondary"), f"Soggetto{participant}")

    def secondary_kinematics_file_path(self, participant: int, movement: int) -> str:
        return os.path.join(self.secondary_subject_root(participant), f"MovimentoAngS{movement}.mat")

    def secondary_timing_file_path(self, participant: int, movement: int) -> str:
        subject_root = self.secondary_subject_root(participant)
        if movement == 9:
            return os.path.join(subject_root, "InizioFineRest12.mat")
        if 1 <= movement <= 8:
            return os.path.join(subject_root, f"InizioFineSteady{movement + 10}.mat")
        raise ValueError("Secondary timing files are only defined for movements 1-9.")

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

    def selection_from_path(self, file_path: str) -> Optional[FileSelection]:
        """Best-effort conversion from a MAT file path back to a file selection."""
        norm_path = os.path.normpath(file_path).replace("\\", "/")

        collected_match = re.search(r"(?:^|/)collected/(?:raw|edited)/P(\d+)M(\d+)(?:_labelled)?\.mat$", norm_path, flags=re.IGNORECASE)
        if collected_match:
            return FileSelection(
                data_type="collected",
                participant=int(collected_match.group(1)),
                movement=int(collected_match.group(2)),
            )

        secondary_match = re.search(
            r"(?:^|/)secondary/(?:raw|edited)/Soggetto(\d+)/Movimento(\d+)(?:_labelled)?\.mat$",
            norm_path,
            flags=re.IGNORECASE,
        )
        if secondary_match:
            return FileSelection(
                data_type="secondary",
                participant=int(secondary_match.group(1)),
                movement=int(secondary_match.group(2)),
            )

        return None

    def labelled_candidate_path(self, file_path: str) -> str:
        """Returns the best-effort labelled companion path for a MAT file path."""
        selection = self.selection_from_path(file_path)
        if selection is not None:
            return self.output_file_path(selection, create_dirs=False)

        norm_path = os.path.normpath(file_path)
        if not norm_path.lower().endswith(".mat"):
            return norm_path
        if norm_path.lower().endswith("_labelled.mat"):
            return norm_path

        base, _ = os.path.splitext(norm_path)
        labelled = base + "_labelled.mat"
        raw_segment = f"{os.sep}raw{os.sep}"
        edited_segment = f"{os.sep}edited{os.sep}"
        if raw_segment in labelled:
            labelled = labelled.replace(raw_segment, edited_segment)
        return labelled

    def load_mat(self, file_path: str, **load_kwargs) -> Dict[str, object]:
        return scipy.io.loadmat(file_path, **load_kwargs)

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

    def iter_file_selections(self, data_type: str, participant_ids: Optional[Sequence[int]] = None) -> List[FileSelection]:
        """Returns available participant/movement selections for a dataset type."""
        participants = self.discover_participants(data_type)
        if participant_ids is not None:
            requested_participants = {int(participant) for participant in participant_ids}
            participants = [participant for participant in participants if participant in requested_participants]

        selections: List[FileSelection] = []
        for participant in participants:
            for movement in self.discover_movements(data_type, participant):
                selections.append(
                    FileSelection(
                        data_type=data_type,
                        participant=participant,
                        movement=movement,
                    )
                )

        return selections

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
