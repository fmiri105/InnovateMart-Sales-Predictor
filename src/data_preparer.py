import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


class SalesDataPreparer:
    """
    Prepare sales DataFrame for creating training and validation datasets compatible with PyTorch Forecasting.

    This class converts raw sales data into TimeSeriesDataSet format with train-validation split based on the specified number of validation days.
    """

    def __init__(
        self,
        df_sales: pd.DataFrame,
        encoder_length=60,
        prediction_length=30,
        validation_days=30,
    ):
        """
        Initialize the SalesDataPreparer with raw data and configuration parameters.

        Args:
            df_sales (pd.DataFrame): Raw sales data.
            encoder_length (int): Length of the encoder (input sequence length).
            prediction_length (int): Length of the prediction horizon (output sequence length).
            validation_days (int): Number of days for validation (last N days used for validation).
        """
        self.df_sales = df_sales.copy()
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.validation_days = validation_days

        # Convert 'Date' column to datetime format
        self.df_sales['Date'] = pd.to_datetime(self.df_sales['Date'])

        # Generate a time index column counting days from the earliest date
        self.df_sales['time_idx'] = (self.df_sales['Date'] - self.df_sales['Date'].min()).dt.days

        # Convert specified categorical columns to 'category' dtype for memory efficiency and modeling
        cat_cols = ['store_id', 'store_size', 'promotion_active', 'is_weekend', 'is_holiday', 'month']
        for col in cat_cols:
            if col in self.df_sales.columns:
                self.df_sales[col] = self.df_sales[col].astype(str).astype('category')

    def prepare(self):
        """
        Prepare training and validation datasets and dataloaders for model training and evaluation.

        Returns:
            training_dataset (TimeSeriesDataSet): Dataset for training.
            validation_dataset (TimeSeriesDataSet): Dataset for validation/prediction.
            train_dataloader (DataLoader): PyTorch DataLoader for training dataset.
            val_dataloader (DataLoader): PyTorch DataLoader for validation dataset.
        """
        max_time_idx = self.df_sales['time_idx'].max()
        training_cutoff = max_time_idx - self.validation_days  # Cutoff day for separating training and validation

        # Filter the dataframe to include data up to the cutoff for training
        training_df = self.df_sales[self.df_sales['time_idx'] <= training_cutoff].copy()

        # Create the training TimeSeriesDataSet object
        training_dataset = TimeSeriesDataSet(
            training_df,
            time_idx='time_idx',
            target='daily_sales',
            group_ids=['store_id'],
            max_encoder_length=self.encoder_length,
            max_prediction_length=self.prediction_length,
            static_categoricals=['store_id', 'store_size'],
            static_reals=['city_population'],
            time_varying_known_categoricals=['promotion_active', 'is_weekend', 'is_holiday', 'month'],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=['daily_sales'],
            target_normalizer=GroupNormalizer(groups=['store_id'], transformation='softplus'),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Construct validation dataset based on all available data for prediction
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            self.df_sales,
            predict=True,
            stop_randomization=True,
        )

        batch_size = 64
        # Create dataloaders for training and validation datasets
        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        print(f"Training dataset length: {len(training_dataset)} samples")
        print(f"Validation dataset length: {len(validation_dataset)} samples")

        return training_dataset, validation_dataset, train_dataloader, val_dataloader
