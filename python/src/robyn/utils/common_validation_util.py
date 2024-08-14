class CommonValidationUtil:
    def __init__(self):
        pass

    def check_is_vector(obj: Any) -> bool:
        if not isinstance(obj, list):
            return False
        return True

    def check_daterange(
        date_min: Optional[datetime],
        date_max: Optional[datetime],
        dates: List[datetime],
    ) -> bool:
        """
        Check if the date range is valid.

        Args:
            date_min (Optional[datetime]): The minimum date.
            date_max (Optional[datetime]): The maximum date.
            dates (List[datetime]): The dates.

        Returns:
            bool: True if the date range is valid, False otherwise.
        """
        if date_min is not None:
            if date_min < min(dates):
                return False
        if date_max is not None:
            if date_max > max(dates):
                return False
        return True

    def format_date(date: Any) -> str:
        """
        Format a date object into a string YYYY-MM-DD.
        """
        if isinstance(date, np.datetime64):
            date = str(np.datetime_as_string(date, unit="D"))
            date = datetime.strptime(date, "%Y-%m-%d")
            return date.strftime("%Y-%m-%d")
        elif isinstance(date, datetime):
            return date.strftime("%Y-%m-%d")
        else:
            raise TypeError("Unsupported date type")

    def validate_instance_types(
        elements: List[Any], expected_type: Type
    ) -> ValidationResult:
        """
        Validates that all elements in the provided list are instances of the specified class type.
        Args:
            elements (List[Any]): List of elements to be checked.
            expected_type (Type): The class type that all elements in the list are expected to be instances of.
        """
        raise NotImplementedError("Not yet implemented")
