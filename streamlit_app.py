import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer


@st.cache_data
def load_data(path="data/simulated_sales_data.csv"):
    """
    Load and preprocess the sales data CSV file.

    Caches the result to optimize app performance by avoiding repeated disk reads.

    Args:
        path (str): File path to load the CSV data from.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with correct datatypes and a time index.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    cat_cols = ["store_id", "store_size", "promotion_active", "is_weekend", "is_holiday", "month"]
    for col in cat_cols:
        df[col] = df[col].astype(str).astype("category")
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days
    return df


def prepare_datasets(df, encoder_length=60, prediction_length=30, validation_days=30):
    """
    Prepare training and validation TimeSeriesDataSet objects from the input DataFrame.

    Splits data by time index for validation, defines static and time-varying features,
    and applies GroupNormalizer for target scaling.

    Args:
        df (pd.DataFrame): Input sales data.
        encoder_length (int): Length of input sequence for encoder.
        prediction_length (int): Length of prediction horizon.
        validation_days (int): Number of days reserved for validation.

    Returns:
        Tuple[TimeSeriesDataSet, TimeSeriesDataSet]: Training and validation datasets.
    """
    max_time_idx = df['time_idx'].max()
    training_cutoff = max_time_idx - validation_days
    training_df = df[df['time_idx'] <= training_cutoff].copy()

    training_dataset = TimeSeriesDataSet(
        training_df,
        time_idx='time_idx',
        target='daily_sales',
        group_ids=['store_id'],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
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

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        predict=True,
        stop_randomization=True,
    )

    return training_dataset, validation_dataset


@st.cache_resource
def load_model():
    """
    Load the trained Temporal Fusion Transformer model from checkpoint.

    Caches the loaded model in Streamlit to avoid reloading on every interaction.

    Returns:
        TemporalFusionTransformer: Loaded model in evaluation mode.
    """
    model = TemporalFusionTransformer.load_from_checkpoint("models/best_model.ckpt", strict=False)
    model.eval()
    return model


def predict_for_store(model, df, store_id, encoder_length=60, prediction_length=30):
    """
    Generate sales predictions for a specific store using the trained model.

    Constructs a TimeSeriesDataSet for the specified store subset and runs model prediction.

    Args:
        model (TemporalFusionTransformer): Trained forecasting model.
        df (pd.DataFrame): Complete sales DataFrame.
        store_id (str): Store identifier for prediction.
        encoder_length (int): Input sequence length used by the model.
        prediction_length (int): Output prediction horizon length.

    Returns:
        tuple: (predictions, actual sales, raw model outputs for interpretability)
    """
    df_store = df[df['store_id'] == store_id].copy()

    validation_dataset_params = dict(
        time_idx='time_idx',
        target='daily_sales',
        group_ids=['store_id'],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=['store_id', 'store_size'],
        static_reals=['city_population'],
        time_varying_known_categoricals=['promotion_active', 'is_weekend', 'is_holiday', 'month'],
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['daily_sales'],
        target_normalizer=GroupNormalizer(groups=['store_id'], transformation='softplus'),
    )

    dataset_store = TimeSeriesDataSet(
        df_store,
        time_idx=validation_dataset_params['time_idx'],
        target=validation_dataset_params['target'],
        group_ids=validation_dataset_params['group_ids'],
        max_encoder_length=validation_dataset_params['max_encoder_length'],
        max_prediction_length=validation_dataset_params['max_prediction_length'],
        static_categoricals=validation_dataset_params['static_categoricals'],
        static_reals=validation_dataset_params['static_reals'],
        time_varying_known_categoricals=validation_dataset_params['time_varying_known_categoricals'],
        time_varying_known_reals=validation_dataset_params['time_varying_known_reals'],
        time_varying_unknown_categoricals=validation_dataset_params['time_varying_unknown_categoricals'],
        time_varying_unknown_reals=validation_dataset_params['time_varying_unknown_reals'],
        target_normalizer=validation_dataset_params['target_normalizer'],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    dataloader = dataset_store.to_dataloader(batch_size=1, train=False, num_workers=3)

    with torch.no_grad():
        # Use raw mode and return_x=True for model interpretability outputs
        raw_predictions, x, *rest = model.predict(dataloader, mode="raw", return_x=True)

    preds_mean = raw_predictions["prediction"].cpu().numpy()

    actuals = []
    for batch in dataloader:
        actual = batch[1][0].cpu().numpy()
        actuals.append(actual.flatten())
    actuals = np.concatenate(actuals)

    predictions = preds_mean[-1, :]

    actuals_plot = actuals[-prediction_length:]

    return predictions, actuals_plot, raw_predictions


def plot_sales_vs_prediction(df, store_id, predictions, actuals, encoder_length=60, prediction_length=30):
    """
    Plot actual sales and predicted sales for a given store over the prediction period.

    Displays results using Matplotlib integrated into Streamlit.

    Args:
        df (pd.DataFrame): Full sales data.
        store_id (str): Store to plot.
        predictions (np.array): Model predicted sales values.
        actuals (np.array): Actual observed sales values.
        encoder_length (int): Length of encoder input used.
        prediction_length (int): Length of prediction horizon.
    """
    df_store = df[df["store_id"] == store_id].copy().reset_index(drop=True)

    total_days = encoder_length + prediction_length
    df_window = df_store.tail(total_days).copy().reset_index()

    plt.figure(figsize=(12, 6))

    # Select exact dates corresponding to the prediction horizon
    pred_dates = df_window['Date'].iloc[encoder_length:encoder_length + prediction_length].reset_index(drop=True)

    actuals_dates = pred_dates
    actuals_values = actuals  # Expecting actuals length to match prediction_length

    # Extract median prediction (index 3 corresponds to median quantile 0.5)
    prediction_to_plot = predictions[:, 3]

    # Plot actual sales for prediction horizon
    plt.plot(actuals_dates, actuals_values, label="Actual Sales (30 days)", marker='o')

    # Plot predicted sales for the same dates
    plt.plot(pred_dates, prediction_to_plot, label="Predicted Sales (30 days)", marker='x')

    plt.title(f"Actual vs Predicted Sales for Store {store_id}")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(plt)
    plt.clf()


def plot_variable_importance(model, raw_predictions):
    """
    Plot variable importance extracted from the Temporal Fusion Transformer model.

    Uses the model's interpret_output method for encoder variables and displays a horizontal bar chart.

    Args:
        model (TemporalFusionTransformer): Trained TFT model.
        raw_predictions (dict): Raw model output including interpretability information.
    """
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    var_imp = interpretation["encoder_variables"]

    if not isinstance(var_imp, pd.DataFrame):
        var_imp = pd.Series(var_imp.cpu().numpy(), index=model.encoder_variables)

    var_imp_sorted = var_imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(var_imp_sorted.index, var_imp_sorted.values)
    ax.set_xlabel("Importance")
    ax.set_title("Variable Importance from TFT Model (Encoder Variables)")
    plt.tight_layout()

    st.pyplot(fig)


def main():
    """
    Streamlit app main function that provides interactive interface for sales forecasting per store.

    Loads data and model, allows store selection, shows historical sales,
    predictions, and variable importance.
    """
    st.title("پیش‌بینی فروش روزانه فروشگاه‌ها با مدل Temporal Fusion Transformer")
    st.write("لطفا فروشگاه را از لیست زیر انتخاب نمایید:")

    df = load_data()
    store_ids = df["store_id"].cat.categories.tolist()
    selected_store = st.selectbox("انتخاب فروشگاه", store_ids)

    training_dataset, validation_dataset = prepare_datasets(df)

    model = load_model()

    st.write(f"فروش تاریخی فروشگاه {selected_store}:")
    df_store = df[df["store_id"] == selected_store].copy()
    st.line_chart(df_store.set_index("Date")["daily_sales"])

    st.write("مشخصات مدل و پیش‌بینی‌ها:")
    predictions, actuals, raw_predictions = predict_for_store(model, df, selected_store)

    plot_sales_vs_prediction(df, selected_store, predictions, actuals)

    st.write("اهمیت متغیرهای ورودی مدل:")
    plot_variable_importance(model, raw_predictions)


if __name__ == "__main__":
    main()
