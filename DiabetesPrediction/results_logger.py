import os
import time
import pandas as pd


def save_results(history, test_loss, test_accuracy, epochs, batch_size, plt):
    os.makedirs("model_results", exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    run_folder = f"model_results/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    graph_filename = f"{run_folder}/accuracy_graph_{timestamp}.png"
    plt.savefig(graph_filename)

    epoch_filename = f"{run_folder}/epoch_results_{timestamp}.csv"

    epoch_df = pd.DataFrame(history.history)
    epoch_df.index = epoch_df.index + 1
    epoch_df.index.name = "Epoch"
    epoch_df.to_csv(epoch_filename)

    train_accuracy = history.history["accuracy"][-1]
    validation_accuracy = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    validation_loss = history.history["val_loss"][-1]

    general_csv = "model_results/general_results.csv"

    new_row = {
        "RunTime": timestamp,
        "Epochs": epochs,
        "BatchSize": batch_size,
        "TrainAccuracy": round(train_accuracy, 4),
        "ValidationAccuracy": round(validation_accuracy, 4),
        "TrainLoss": round(train_loss, 4),
        "ValidationLoss": round(validation_loss, 4),
        "TestAccuracy": round(test_accuracy, 4),
        "TestLoss": round(test_loss, 4),
        "RunFolder": run_folder,
        "GraphFile": graph_filename,
        "EpochFile": epoch_filename
    }

    if os.path.exists(general_csv):
        df_results = pd.read_csv(general_csv)
    else:
        df_results = pd.DataFrame()

    df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)

    column_order = [
        "RunTime",
        "Epochs",
        "BatchSize",
        "TrainAccuracy",
        "ValidationAccuracy",
        "TrainLoss",
        "ValidationLoss",
        "TestAccuracy",
        "TestLoss",
        "RunFolder",
        "GraphFile",
        "EpochFile"
    ]

    df_results = df_results.reindex(columns=column_order)
    df_results.to_csv(general_csv, index=False)