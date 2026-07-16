import argparse
import csv
import re
import shutil
from pathlib import Path

import dill as pickle
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from cadelac.learning.data_scripts.utils import load_dataset
from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN


EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+):"
    r".*?Time/Epoch\s*=\s*([0-9.eE+-]+)s,"
    r"\s*Loss\s*=\s*([0-9.eE+-]+),"
    r"\s*Inv Dyn\s*=\s*([0-9.eE+-]+)"
    r"\s*±\s*([0-9.eE+-]+),"
    r"\s*Power Con\s*=\s*([0-9.eE+-]+)"
    r"\s*±\s*([0-9.eE+-]+)"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot the context-aware LSTM torque MLP."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained .torch checkpoint."
    )

    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the training log."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which all evaluation artifacts are stored."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size used during model evaluation."
    )

    return parser.parse_args()


def parse_training_log(log_path):
    epochs = []
    time_per_epoch = []
    losses = []
    inverse_dynamics = []
    inverse_dynamics_interval = []
    power = []
    power_interval = []

    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    for match in EPOCH_PATTERN.finditer(log_text):
        epochs.append(int(match.group(1)))
        time_per_epoch.append(float(match.group(2)))
        losses.append(float(match.group(3)))
        inverse_dynamics.append(float(match.group(4)))
        inverse_dynamics_interval.append(float(match.group(5)))
        power.append(float(match.group(6)))
        power_interval.append(float(match.group(7)))

    if not epochs:
        raise RuntimeError(
            f"No epoch entries could be parsed from log file: {log_path}"
        )

    return {
        "epoch": np.asarray(epochs),
        "time_per_epoch": np.asarray(time_per_epoch),
        "loss": np.asarray(losses),
        "inverse_dynamics": np.asarray(inverse_dynamics),
        "inverse_dynamics_interval": np.asarray(
            inverse_dynamics_interval
        ),
        "power": np.asarray(power),
        "power_interval": np.asarray(power_interval)
    }


def load_test_data(learning_dir, hist_length, n_dof):
    dataset_name = (
        "exo_hip_knee_delan_2dof_left_all_trials_context"
    )

    dataset_path = (
        learning_dir
        / "datasets"
        / "panda"
        / f"{dataset_name}.pkl"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset was not found: {dataset_path}"
        )

    with dataset_path.open("rb") as file:
        raw_data = pickle.load(file)

    test_labels = [
        label
        for label in raw_data["labels"]
        if "BT24" in label
    ]

    train_data, test_data, divider, dt_mean = load_dataset(
        filename=str(dataset_path),
        test_label=test_labels,
        full_model=False,
        sample_offset=1,
        dataset_use=1.0,
        n_dof=n_dof,
        hist_length=hist_length,
        hist_labels=["qp", "qv", "tau", "diff_tau"],
        add_noise=False
    )

    (
        train_labels,
        train_qp,
        train_qv,
        train_qa,
        train_tau,
        train_hist_qp,
        train_hist_qv,
        train_hist_tau,
        train_hist_diff_tau
    ) = train_data

    (
        loaded_test_labels,
        test_qp,
        test_qv,
        test_qa,
        test_tau,
        test_m,
        test_c,
        test_g,
        test_hist_qp,
        test_hist_qv,
        test_hist_tau,
        test_hist_diff_tau
    ) = test_data

    test_lstm_input = np.concatenate(
        (
            test_hist_qp,
            test_hist_qv,
            test_hist_tau
        ),
        axis=-1
    )

    return {
        "train_tau": train_tau,
        "test_labels": loaded_test_labels,
        "test_qp": test_qp,
        "test_qv": test_qv,
        "test_qa": test_qa,
        "test_tau": test_tau,
        "test_lstm_input": test_lstm_input,
        "divider": np.asarray(divider, dtype=int),
        "dt_mean": dt_mean
    }


def load_model(checkpoint_path, n_dof):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False
    )

    hyper = checkpoint["hyper"]

    model = ContextAwareDeLaN(
        n_dof,
        **hyper
    )

    model.load_state_dict(checkpoint["state_dict"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()

    model.eval()

    return model, hyper, device, checkpoint


def predict_in_batches(
    model,
    device,
    q,
    qd,
    qdd,
    lstm_input,
    batch_size
):
    predicted_torque = []
    predicted_power = []

    number_of_samples = q.shape[0]

    with torch.no_grad():
        for start in range(0, number_of_samples, batch_size):
            end = min(start + batch_size, number_of_samples)

            q_batch = torch.from_numpy(
                q[start:end]
            ).float().to(device)

            qd_batch = torch.from_numpy(
                qd[start:end]
            ).float().to(device)

            qdd_batch = torch.from_numpy(
                qdd[start:end]
            ).float().to(device)

            history_batch = torch.from_numpy(
                lstm_input[start:end]
            ).float().to(device)

            tau_batch, power_batch = model(
                q_batch,
                qd_batch,
                qdd_batch,
                history_batch
            )

            predicted_torque.append(
                tau_batch.detach().cpu().numpy()
            )

            predicted_power.append(
                power_batch.detach().cpu().numpy()
            )

    return (
        np.concatenate(predicted_torque, axis=0),
        np.concatenate(predicted_power, axis=0)
    )


def calculate_metrics(target, prediction):
    error = prediction - target

    mse = np.mean(error ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error), axis=0)

    target_mean = np.mean(target, axis=0)
    residual_sum = np.sum(error ** 2, axis=0)
    total_sum = np.sum(
        (target - target_mean) ** 2,
        axis=0
    )

    r_squared = np.where(
        total_sum > 0,
        1.0 - residual_sum / total_sum,
        np.nan
    )

    return {
        "error": error,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared
    }


def shortened_label(label):
    if "__" in label:
        return label.split("__", maxsplit=1)[1]

    return label


def add_run_boundaries(axis, divider):
    for position in divider[1:-1]:
        axis.axvline(
            position,
            linestyle="--",
            linewidth=0.6,
            alpha=0.5
        )


def add_run_labels(axis, labels, divider):
    centers = (
        divider[:-1] + divider[1:]
    ) / 2.0

    axis.set_xticks(centers)

    axis.set_xticklabels(
        [shortened_label(label) for label in labels],
        rotation=90,
        fontsize=6
    )


def save_training_plots(training, output_dir):
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.semilogy(
        training["epoch"],
        training["loss"],
        label="Training loss"
    )

    axis.semilogy(
        training["epoch"],
        training["inverse_dynamics"],
        label="Torque loss"
    )

    axis.semilogy(
        training["epoch"],
        training["power"],
        label="Power error"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("LSTM + MLP training curves")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_dir / "01_training_loss.png",
        dpi=200
    )

    figure.savefig(
        output_dir / "01_training_loss.pdf"
    )

    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.plot(
        training["epoch"],
        training["time_per_epoch"]
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Time per epoch [s]")
    axis.set_title("Training time per epoch")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_dir / "02_training_time.png",
        dpi=200
    )

    plt.close(figure)


def save_torque_overview(
    target,
    prediction,
    labels,
    divider,
    output_dir
):
    n_dof = target.shape[1]

    figure, axes = plt.subplots(
        n_dof,
        1,
        figsize=(16, 4 * n_dof),
        sharex=True
    )

    axes = np.atleast_1d(axes)

    joint_names = ["Hip", "Knee"]

    for joint in range(n_dof):
        axis = axes[joint]

        axis.plot(
            target[:, joint],
            label="Ground truth",
            linewidth=1.0
        )

        axis.plot(
            prediction[:, joint],
            label="LSTM + MLP",
            linewidth=0.8,
            alpha=0.8
        )

        joint_name = (
            joint_names[joint]
            if joint < len(joint_names)
            else f"Joint {joint}"
        )

        axis.set_ylabel(
            f"{joint_name} residual torque [Nm]"
        )

        axis.grid(True, alpha=0.25)
        add_run_boundaries(axis, divider)

    axes[0].legend()
    axes[-1].set_xlabel("Test sample")
    add_run_labels(axes[-1], labels, divider)

    figure.suptitle(
        "Residual torque prediction on unseen subject BT24"
    )

    figure.tight_layout()

    figure.savefig(
        output_dir / "03_torque_prediction_overview.png",
        dpi=200
    )

    figure.savefig(
        output_dir / "03_torque_prediction_overview.pdf"
    )

    plt.close(figure)


def save_error_overview(
    error,
    labels,
    divider,
    output_dir
):
    n_dof = error.shape[1]

    figure, axes = plt.subplots(
        n_dof,
        1,
        figsize=(16, 4 * n_dof),
        sharex=True
    )

    axes = np.atleast_1d(axes)

    joint_names = ["Hip", "Knee"]

    for joint in range(n_dof):
        axis = axes[joint]

        axis.plot(
            error[:, joint],
            linewidth=0.8
        )

        axis.axhline(
            0.0,
            linestyle="--",
            linewidth=0.8
        )

        joint_name = (
            joint_names[joint]
            if joint < len(joint_names)
            else f"Joint {joint}"
        )

        axis.set_ylabel(
            f"{joint_name} error [Nm]"
        )

        axis.grid(True, alpha=0.25)
        add_run_boundaries(axis, divider)

    axes[-1].set_xlabel("Test sample")
    add_run_labels(axes[-1], labels, divider)

    figure.suptitle(
        "Prediction error: predicted torque minus ground truth"
    )

    figure.tight_layout()

    figure.savefig(
        output_dir / "04_torque_error_overview.png",
        dpi=200
    )

    figure.savefig(
        output_dir / "04_torque_error_overview.pdf"
    )

    plt.close(figure)


def save_scatter_plots(target, prediction, output_dir):
    n_dof = target.shape[1]
    joint_names = ["hip", "knee"]

    max_points = 15000

    if target.shape[0] > max_points:
        indices = np.linspace(
            0,
            target.shape[0] - 1,
            max_points,
            dtype=int
        )
    else:
        indices = np.arange(target.shape[0])

    for joint in range(n_dof):
        figure, axis = plt.subplots(figsize=(6, 6))

        target_joint = target[indices, joint]
        prediction_joint = prediction[indices, joint]

        low = min(
            np.min(target_joint),
            np.min(prediction_joint)
        )

        high = max(
            np.max(target_joint),
            np.max(prediction_joint)
        )

        axis.scatter(
            target_joint,
            prediction_joint,
            s=5,
            alpha=0.25
        )

        axis.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=1.0,
            label="Ideal prediction"
        )

        axis.set_xlabel("Ground truth torque [Nm]")
        axis.set_ylabel("Predicted torque [Nm]")

        joint_name = (
            joint_names[joint]
            if joint < len(joint_names)
            else f"joint_{joint}"
        )

        axis.set_title(
            f"Predicted versus ground truth: {joint_name}"
        )

        axis.grid(True, alpha=0.25)
        axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_dir
            / f"05_scatter_{joint_name}.png",
            dpi=200
        )

        plt.close(figure)


def save_metric_plot(metrics, output_dir):
    joint_names = ["Hip", "Knee"]
    x_positions = np.arange(len(metrics["rmse"]))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        x_positions - width / 2,
        metrics["rmse"],
        width,
        label="RMSE"
    )

    axis.bar(
        x_positions + width / 2,
        metrics["mae"],
        width,
        label="MAE"
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            joint_names[index]
            if index < len(joint_names)
            else f"Joint {index}"
            for index in range(len(x_positions))
        ]
    )

    axis.set_ylabel("Torque error [Nm]")
    axis.set_title("Torque prediction error per joint")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_dir / "06_metrics_per_joint.png",
        dpi=200
    )

    plt.close(figure)


def safe_filename(label):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        label
    )


def save_per_run_plots(
    target,
    prediction,
    labels,
    divider,
    output_dir
):
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    for run_index, label in enumerate(labels):
        start = divider[run_index]
        end = divider[run_index + 1]

        run_target = target[start:end]
        run_prediction = prediction[start:end]
        run_error = run_prediction - run_target

        figure, axes = plt.subplots(
            2,
            2,
            figsize=(14, 8),
            sharex="col"
        )

        axes[0, 0].plot(
            run_target[:, 0],
            label="Ground truth"
        )

        axes[0, 0].plot(
            run_prediction[:, 0],
            label="LSTM + MLP",
            alpha=0.8
        )

        axes[0, 0].set_title("Hip residual torque")
        axes[0, 0].set_ylabel("Torque [Nm]")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.25)

        axes[1, 0].plot(run_error[:, 0])
        axes[1, 0].axhline(0.0, linestyle="--")
        axes[1, 0].set_title("Hip error")
        axes[1, 0].set_xlabel("Sample")
        axes[1, 0].set_ylabel("Error [Nm]")
        axes[1, 0].grid(True, alpha=0.25)

        axes[0, 1].plot(
            run_target[:, 1],
            label="Ground truth"
        )

        axes[0, 1].plot(
            run_prediction[:, 1],
            label="LSTM + MLP",
            alpha=0.8
        )

        axes[0, 1].set_title("Knee residual torque")
        axes[0, 1].set_ylabel("Torque [Nm]")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.25)

        axes[1, 1].plot(run_error[:, 1])
        axes[1, 1].axhline(0.0, linestyle="--")
        axes[1, 1].set_title("Knee error")
        axes[1, 1].set_xlabel("Sample")
        axes[1, 1].set_ylabel("Error [Nm]")
        axes[1, 1].grid(True, alpha=0.25)

        figure.suptitle(label)
        figure.tight_layout()

        figure.savefig(
            per_run_dir / f"{safe_filename(label)}.png",
            dpi=160
        )

        plt.close(figure)


def save_metrics_csv(metrics, output_path):
    joint_names = ["hip", "knee"]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            ["joint", "mse", "rmse", "mae", "r_squared"]
        )

        for joint in range(len(metrics["mse"])):
            name = (
                joint_names[joint]
                if joint < len(joint_names)
                else f"joint_{joint}"
            )

            writer.writerow(
                [
                    name,
                    metrics["mse"][joint],
                    metrics["rmse"][joint],
                    metrics["mae"][joint],
                    metrics["r_squared"][joint]
                ]
            )


def main():
    args = parse_arguments()

    checkpoint_path = args.checkpoint.resolve()
    log_path = args.log.resolve()
    output_dir = args.output.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    if not log_path.is_file():
        raise FileNotFoundError(
            f"Training log does not exist: {log_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    learning_dir = Path(__file__).resolve().parent
    n_dof = 2

    training = parse_training_log(log_path)

    model, hyper, device, checkpoint = load_model(
        checkpoint_path,
        n_dof
    )

    hist_length = int(hyper["hist_length"])

    data = load_test_data(
        learning_dir,
        hist_length,
        n_dof
    )

    predicted_tau, predicted_power = predict_in_batches(
        model=model,
        device=device,
        q=data["test_qp"],
        qd=data["test_qv"],
        qdd=data["test_qa"],
        lstm_input=data["test_lstm_input"],
        batch_size=args.batch_size
    )

    metrics = calculate_metrics(
        data["test_tau"],
        predicted_tau
    )

    torque_variance = np.var(
        data["train_tau"],
        axis=0
    )

    normalized_torque_mse = (
        np.sum(
            metrics["error"] ** 2
            / torque_variance
        )
        / data["test_tau"].shape[0]
    )

    save_training_plots(
        training,
        output_dir
    )

    save_torque_overview(
        data["test_tau"],
        predicted_tau,
        data["test_labels"],
        data["divider"],
        output_dir
    )

    save_error_overview(
        metrics["error"],
        data["test_labels"],
        data["divider"],
        output_dir
    )

    save_scatter_plots(
        data["test_tau"],
        predicted_tau,
        output_dir
    )

    save_metric_plot(
        metrics,
        output_dir
    )

    save_per_run_plots(
        data["test_tau"],
        predicted_tau,
        data["test_labels"],
        data["divider"],
        output_dir
    )

    save_metrics_csv(
        metrics,
        output_dir / "metrics.csv"
    )

    np.savez_compressed(
        output_dir / "predictions.npz",
        target_tau=data["test_tau"],
        predicted_tau=predicted_tau,
        prediction_error=metrics["error"],
        predicted_power=predicted_power,
        test_q=data["test_qp"],
        test_qd=data["test_qv"],
        test_qdd=data["test_qa"],
        divider=data["divider"],
        labels=np.asarray(
            data["test_labels"],
            dtype=object
        )
    )

    summary_lines = [
        "Context-aware LSTM + MLP evaluation",
        "===================================",
        f"Checkpoint: {checkpoint_path}",
        f"Training log: {log_path}",
        f"Evaluation device: {device}",
        f"Checkpoint epoch: {checkpoint.get('epoch')}",
        f"Number of test samples: {data['test_tau'].shape[0]}",
        f"Median timestep: {data['dt_mean']:.8f} s",
        f"Normalized total torque MSE: {normalized_torque_mse:.8e}",
        "",
        "Per-joint metrics:"
    ]

    for joint in range(n_dof):
        summary_lines.extend(
            [
                f"Joint {joint}:",
                f"  MSE:  {metrics['mse'][joint]:.8e}",
                f"  RMSE: {metrics['rmse'][joint]:.8e}",
                f"  MAE:  {metrics['mae'][joint]:.8e}",
                f"  R2:   {metrics['r_squared'][joint]:.8e}"
            ]
        )

    summary_lines.extend(
        [
            "",
            "Important:",
            "The direct torque MLP does not provide a physically",
            "identified decomposition into inertia, Coriolis and",
            "gravity torques. Only the total residual torque",
            "prediction is interpreted as the primary model output."
        ]
    )

    summary_path = output_dir / "summary.txt"

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8"
    )

    shutil.copy2(
        checkpoint_path,
        output_dir / checkpoint_path.name
    )

    shutil.copy2(
        log_path,
        output_dir / log_path.name
    )

    shutil.copy2(
        learning_dir
        / "models"
        / "context_aware_delan.py",
        output_dir / "context_aware_delan.py"
    )

    shutil.copy2(
        learning_dir / "train_panda.py",
        output_dir / "train_panda.py"
    )

    shutil.copy2(
        Path(__file__).resolve(),
        output_dir / "plot_mlp_results.py"
    )

    print("\nEvaluation completed.")
    print(f"Output directory: {output_dir}")
    print(f"Normalized torque MSE: {normalized_torque_mse:.8e}")

    for joint in range(n_dof):
        print(
            f"Joint {joint}: "
            f"RMSE={metrics['rmse'][joint]:.6f}, "
            f"MAE={metrics['mae'][joint]:.6f}, "
            f"R2={metrics['r_squared'][joint]:.6f}"
        )


if __name__ == "__main__":
    main()
