import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


class SalesModelTrainer:
    """
    Trainer class for Temporal Fusion Transformer (TFT) model on sales time series data.

    Handles model creation, training configuration, checkpointing, and saving best model.
    """

    def __init__(self, training_dataset, train_dataloader, val_dataloader, model_dir="models"):
        """
        Initialize the trainer with dataset, dataloaders, and model saving directory.

        Args:
            training_dataset: TimeSeriesDataSet for model input shapes and metadata.
            train_dataloader: DataLoader for training data batching.
            val_dataloader: DataLoader for validation data batching.
            model_dir (str): Directory path to save trained model checkpoints.
        """
        self.training_dataset = training_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)  # Create directory if not existing

    def train(self, max_epochs=20):
        """
        Train the Temporal Fusion Transformer model using PyTorch Lightning.

        Args:
            max_epochs (int): Maximum number of training epochs.

        Returns:
            model: Trained TemporalFusionTransformer model instance.
            trainer: PyTorch Lightning Trainer instance.
        """
        # Instantiate TFT model using dataset metadata and chosen hyperparameters
        model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=0.02,
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=7,  # Number of quantiles for probabilistic forecasting
            loss=QuantileLoss(),  # Loss function for quantile regression
            lstm_layers=1,
            reduce_on_plateau_patience=4,
            log_interval=10,
        )

        # Setup PyTorch Lightning trainer with callbacks, logging, and resources config
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            logger=TensorBoardLogger("lightning_logs"),
            callbacks=[
                ModelCheckpoint(
                    dirpath=self.model_dir,
                    filename="best_model",
                    monitor="val_loss",  # Monitor validation loss for checkpointing
                    mode="min",
                    save_top_k=1,  # Keep only the best model checkpoint
                )
            ],
            gradient_clip_val=0.1,
            limit_val_batches=1.0,  # Use full validation dataset on each epoch
        )

        print(f"Starting training for {max_epochs} epochs...")

        # Train the model using train and validation dataloaders
        trainer.fit(
            model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )

        # Save the best model checkpoint and notify user
        trainer.save_checkpoint(os.path.join(self.model_dir, "best_model.ckpt"))

        print(f"Best model saved at '{os.path.join(self.model_dir, 'best_model.ckpt')}'")

        return model, trainer
