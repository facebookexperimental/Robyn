from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.data.validation.validation import Validation, ValidationResult


@dataclass
class MMMDataCollectionValidation(Validation):
    def __init__(self, mmmdata_collection: MMMDataCollection) -> None:
        self.mmmdata_collection: MMMDaMMMDataCollectionta = mmmdata_collection

    def check_mmmdata_collect(
        mmmdata_collection: MMMDataCollection,
    ) -> ValidationResult:
        """
        Check if the MMMDataCollection has all of the required fields populated.
        """
        raise NotImplementedError("Not yet implemented")
