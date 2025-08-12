from src.data_simulator import SalesDataSimulator
from src.data_preparer import SalesDataPreparer
from src.model_trainer import SalesModelTrainer
import os


def main_train():
    """
    Execute the full pipeline for sales forecasting model training:
    1. Simulate synthetic sales data and save to disk.
    2. Prepare the data into training and validation datasets and dataloaders.
    3. Train the Temporal Fusion Transformer model and save the best checkpoint.
    """
    # Step 1: Simulate synthetic sales data
    simulator = SalesDataSimulator()
    data = simulator.simulate()
    simulator.save(data)

    # Step 2: Prepare datasets and dataloaders
    preparer = SalesDataPreparer(data)
    training_dataset, validation_dataset, train_dataloader, val_dataloader = preparer.prepare()

    # Step 3: Initialize trainer, train model, and save checkpoint
    trainer = SalesModelTrainer(training_dataset, train_dataloader, val_dataloader)
    model, pl_trainer = trainer.train(max_epochs=15)

    print("Model training completed and saved to 'models/' directory")


if __name__ == "__main__":
    # Run the training pipeline when script is executed directly
    main_train()
