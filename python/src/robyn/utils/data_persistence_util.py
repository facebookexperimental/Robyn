import json
from typing import Dict, Any
from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.modeling.entities.modeloutput_collection import ModelOutputCollection

class DataPersistenceUtil:
    @staticmethod
    def write_mmm_data_collection_to_json(data: MMMDataCollection, file_path: str) -> None:
        """
        Writes an MMMDataCollection object to a JSON file.

        Args:
            data: The MMMDataCollection object to be written.
            file_path: The path to the JSON file where the data will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(data.to_dict(), f, indent=2)

    @staticmethod
    def read_mmm_data_collection_from_json(file_path: str) -> MMMDataCollection:
        """
        Reads an MMMDataCollection object from a JSON file.

        Args:
            file_path: The path to the JSON file containing the data.

        Returns:
            An MMMDataCollection object reconstructed from the JSON data.
        """
        with open(file_path, 'r') as f:
            data_dict: Dict[str, Any] = json.load(f)
        return MMMDataCollection.from_dict(data_dict)

    @staticmethod
    def write_model_output_collection_to_json(data: ModelOutputCollection, file_path: str) -> None:
        """
        Writes a ModelOutputCollection object to a JSON file.

        Args:
            data: The ModelOutputCollection object to be written.
            file_path: The path to the JSON file where the data will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(data.to_dict(), f, indent=2)

    @staticmethod
    def read_model_output_collection_from_json(file_path: str) -> ModelOutputCollection:
        """
        Reads a ModelOutputCollection object from a JSON file.

        Args:
            file_path: The path to the JSON file containing the data.

        Returns:
            A ModelOutputCollection object reconstructed from the JSON data.
        """
        with open(file_path, 'r') as f:
            data_dict: Dict[str, Any] = json.load(f)
        return ModelOutputCollection.from_dict(data_dict)
