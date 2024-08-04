class DataIngestion:
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {
            'facebook_ads': FacebookAdsData(),
            'google_ads': GoogleAdsData(),
            'tv': TVData(),
            'sales': SalesData()
        }

    def ingest_data(self, source: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Ingest data from a specified source for a given date range.

        Args:
            source (str): The name of the data source.
            start_date (str): The start date for data ingestion (format: 'YYYY-MM-DD').
            end_date (str): The end date for data ingestion (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: The ingested data as a pandas DataFrame.

        Raises:
            ValueError: If the specified source is not supported.
        """
        if source not in self.data_sources:
            raise ValueError(f"Unsupported data source: {source}")

        return self.data_sources[source].ingest(start_date, end_date)

    def ingest_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Ingest data from all available sources for a given date range.

        Args:
            start_date (str): The start date for data ingestion (format: 'YYYY-MM-DD').
            end_date (str): The end date for data ingestion (format: 'YYYY-MM-DD').

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing ingested data for each source.
        """
        return {source: self.ingest_data(source, start_date, end_date) 
                for source in self.data_sources}

    def add_data_source(self, name: str, source: DataSource) -> None:
        """
        Add a new data source to the ingestion system.

        Args:
            name (str): The name of the new data source.
            source (DataSource): An instance of a DataSource subclass.

        Raises:
            ValueError: If a data source with the given name already exists.
        """
        if name in self.data_sources:
            raise ValueError(f"Data source '{name}' already exists")
        self.data_sources[name] = source

    def remove_data_source(self, name: str) -> None:
        """
        Remove a data source from the ingestion system.

        Args:
            name (str): The name of the data source to remove.

        Raises:
            ValueError: If the specified data source does not exist.
        """
        if name not in self.data_sources:
            raise ValueError(f"Data source '{name}' does not exist")
        del self.data_sources[name]

    def get_available_sources(self) -> List[str]:
        """
        Get a list of all available data sources.

        Returns:
            List[str]: A list of names of available data sources.
        """
        return list(self.data_sources.keys())
