import argparse
import os
import time
from pathlib import Path

import dill as pickle
import matplotlib as mp
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cadelac.learning.data_scripts.replay_memory import PyTorchReplayMemory
from cadelac.learning.data_scripts.utils import init_env, load_dataset, plot_torques
from cadelac.learning.models.context_aware_delan import ContextAwareDeLaN


def evaluate_torque_model(
    model,
    qp,
    qv,
    qa,
    tau,
    lstm_history,
    norm_tau,
    hist_length,
    batch_size=8192,
    context_sample_size=4096,
):
    """Evaluate normalized and raw torque MSE without storing all predictions."""
    model.eval()

    n_samples = int(qp.shape[0])
    n_dof = int(tau.shape[-1])

    normalized_joint_sum = torch.zeros(n_dof, dtype=torch.float64)
    raw_joint_sum = torch.zeros(n_dof, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)

            q_batch = torch.from_numpy(qp[start:stop]).float().to(model.device)
            qd_batch = torch.from_numpy(qv[start:stop]).float().to(model.device)
            qdd_batch = torch.from_numpy(qa[start:stop]).float().to(model.device)
            tau_batch = torch.from_numpy(tau[start:stop]).float().to(model.device)

            if hist_length == 0:
                tau_prediction, _ = model(q_batch, qd_batch, qdd_batch)
            else:
                history_batch = torch.from_numpy(
                    lstm_history[start:stop]
                ).float().to(model.device)
                tau_prediction, _ = model(
                    q_batch,
                    qd_batch,
                    qdd_batch,
                    history_batch,
                )

            squared_error = (tau_prediction - tau_batch) ** 2
            normalized_error = squared_error / norm_tau

            normalized_joint_sum += normalized_error.sum(dim=0).double().cpu()
            raw_joint_sum += squared_error.sum(dim=0).double().cpu()

        context_values = None
        if hist_length > 0:
            context_stop = min(context_sample_size, n_samples)
            context_history = torch.from_numpy(
                lstm_history[:context_stop]
            ).float().to(model.device)
            context_values = model.lstm(context_history).detach().cpu()

    model.train()

    normalized_joint_mse = normalized_joint_sum / float(n_samples)
    raw_joint_mse = raw_joint_sum / float(n_samples)

    return {
        "normalized_total_mse": float(normalized_joint_mse.sum().item()),
        "normalized_joint_mse": normalized_joint_mse.numpy(),
        "raw_total_mse": float(raw_joint_mse.sum().item()),
        "raw_joint_mse": raw_joint_mse.numpy(),
        "context_values": context_values,
    }


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Command-line arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        nargs=1,
        type=int,
        required=False,
        default=[True],
        help="Train using CUDA.",
    )
    parser.add_argument(
        "-i",
        nargs=1,
        type=int,
        required=False,
        default=[0],
        help="CUDA device id.",
    )
    parser.add_argument(
        "-s",
        nargs=1,
        type=int,
        required=False,
        default=[0],
        help="Random seed.",
    )
    parser.add_argument(
        "-r",
        nargs=1,
        type=int,
        required=False,
        default=[1],
        help="Render the final torque figure.",
    )
    parser.add_argument(
        "-l",
        nargs=1,
        type=int,
        required=False,
        default=[0],
        help="Load an existing model. Use 0 for a new TensorBoard run.",
    )
    parser.add_argument(
        "-m",
        nargs=1,
        type=int,
        required=False,
        default=[1],
        help="Save the trained model.",
    )
    parser.add_argument(
        "-f",
        nargs=1,
        type=int,
        required=False,
        default=[0],
        help="Learn the full robot model.",
    )
    parser.add_argument(
        "--tb-eval-period",
        type=int,
        default=50,
        help="Evaluate BT24 and write validation metrics every N epochs.",
    )
    parser.add_argument(
        "--tb-histogram-period",
        type=int,
        default=100,
        help="Write model parameter histograms every N epochs.",
    )
    parser.add_argument(
        "--tb-root",
        type=str,
        default=None,
        help="Optional TensorBoard root directory. Default: <repo>/runs.",
    )
    parser.add_argument(
        "--tb-eval-batch-size",
        type=int,
        default=8192,
        help="Batch size used for BT24 TensorBoard evaluation.",
    )

    args = parser.parse_args()
    seed, cuda, render, load_model, save_model, full_model = init_env(args)

    if args.tb_eval_period <= 0:
        raise ValueError("--tb-eval-period must be greater than zero.")
    if args.tb_histogram_period <= 0:
        raise ValueError("--tb-histogram-period must be greater than zero.")
    if args.tb_eval_batch_size <= 0:
        raise ValueError("--tb-eval-batch-size must be greater than zero.")

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------
    nn_id = "ContextAware"
    nn_type = ContextAwareDeLaN

    dataset_use = 1.0
    minibatch = 1024
    loss_power = False

    n_dof = 2
    flag_normalize_tau = True
    sample_offset = 1
    save_checkpoint_model = True
    checkpoint_period = 500
    console_log_period = 50
    add_noise_to_load_data = False

    learning_dir = Path(__file__).resolve().parent
    repo_root = learning_dir.parents[1]

    # ------------------------------------------------------------------
    # Context history: q and qdot only, no torque history
    # ------------------------------------------------------------------
    hist_length = 15 if (nn_id == "ContextAware" and not full_model) else 0

    # The loader still returns torque histories so its return signature remains
    # compatible with the existing data pipeline. They are not passed to the LSTM.
    hist_labels = ["qp", "qv", "tau", "diff_tau"]

    n_lstm_input = n_dof * 2
    n_lstm_hidden = 10
    n_lstm_output = 10
    n_lstm_depth = 5

    if full_model:
        dataset_name = "exo_hip_knee_delan_2dof_left_all_trials"
        model_type_folder = "full_model/panda/" + nn_id
    else:
        dataset_name = "exo_hip_knee_delan_2dof_left_all_trials_context"
        model_type_folder = "res_model/panda/" + nn_id

    dataset_path = learning_dir / "datasets" / "panda" / f"{dataset_name}.pkl"

    # ------------------------------------------------------------------
    # Subject-wise split: BT24 is evaluation only
    # ------------------------------------------------------------------
    with dataset_path.open("rb") as dataset_file:
        dataset_metadata = pickle.load(dataset_file)

    test_label = [
        label
        for label in dataset_metadata["labels"]
        if "BT24" in label
    ]

    if not test_label:
        raise RuntimeError(
            f"No BT24 labels were found in dataset: {dataset_path}"
        )

    print("Subject-wise test labels:", test_label)

    train_data, test_data, divider, dt_mean = load_dataset(
        filename=str(dataset_path),
        test_label=test_label,
        full_model=full_model,
        sample_offset=sample_offset,
        dataset_use=dataset_use,
        n_dof=n_dof,
        hist_length=hist_length,
        hist_labels=hist_labels,
        add_noise=add_noise_to_load_data,
    )

    if hist_length == 0:
        (
            train_labels,
            train_qp,
            train_qv,
            train_qa,
            train_tau,
        ) = train_data

        (
            test_labels,
            test_qp,
            test_qv,
            test_qa,
            test_tau,
            test_m,
            test_c,
            test_g,
        ) = test_data

        n_enc_input = 1
        train_lstm_input = np.ones_like(train_qp)
        test_lstm_input = np.ones_like(test_qp)
    else:
        (
            train_labels,
            train_qp,
            train_qv,
            train_qa,
            train_tau,
            train_hist_qp,
            train_hist_qv,
            train_hist_tau,
            train_hist_diff_tau_nom,
        ) = train_data

        (
            test_labels,
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
            test_hist_diff_tau_nom,
        ) = test_data

        # LSTM input shape:
        # [sample, history timestep, q_hip, q_knee, qdot_hip, qdot_knee]
        train_lstm_input = np.concatenate(
            (train_hist_qp, train_hist_qv),
            axis=-1,
        )
        test_lstm_input = np.concatenate(
            (test_hist_qp, test_hist_qv),
            axis=-1,
        )

        assert train_lstm_input.shape[-1] == n_lstm_input, (
            f"Expected training LSTM input dimension {n_lstm_input}, "
            f"got {train_lstm_input.shape[-1]}."
        )
        assert test_lstm_input.shape[-1] == n_lstm_input, (
            f"Expected test LSTM input dimension {n_lstm_input}, "
            f"got {test_lstm_input.shape[-1]}."
        )

        n_enc_input = n_lstm_output

    assert not any(
        "BT24" in label for label in train_labels
    ), "BT24 leaked into training labels."
    assert all(
        "BT24" in label for label in test_labels
    ), "All test labels are expected to belong to BT24."

    print("\n\n################################################")
    print("Runs:")
    print(f"   Test Runs = {test_labels}")
    print(f"  Train Runs = {train_labels}")
    print(f"# Training Samples = {int(train_qp.shape[0]):05d}")
    print(f"# BT24 Samples = {int(test_qp.shape[0]):05d}")
    print(f"Median timestep = {dt_mean:.6e}s")

    # ------------------------------------------------------------------
    # Model hyperparameters: fixed to 1000 epochs
    # ------------------------------------------------------------------    
    hyper = {
        "diagonal_epsilon": 0.1,
        "activation": "Tanh",
        "net_arch_inertia": [30, 20],
        "net_arch_pot": [30, 20],
        "net_arch_mlp": [64, 64, 64, 64],
        "b_init": 1.0e-4,
        "b_diag_init": 0.001,
        "w_init": "xavier_normal",
        "gain_hidden": np.sqrt(2.0),
        "gain_output": 0.1,
        "n_minibatch": minibatch,
        "learning_rate": 5.0e-4,
        "weight_decay": 1.0e-4,
        "init_tf": True,
        "n_enc_input": n_enc_input,
        "n_lstm_hidden": n_lstm_hidden,
        "n_lstm_input": n_lstm_input,
        "n_lstm_depth": n_lstm_depth,
        "hist_length": hist_length,
        "act_ld": "Softplus",
        "max_epoch": 1000,
    }
    mlp_architecture_tag = "x".join(
        str(neurons)
        for neurons in hyper["net_arch_mlp"]
    )

    model_name = (
        f"mlp_lstm_notau_tensorboard_"
        f"mlp_{mlp_architecture_tag}_"
        f"epochs_{hyper['max_epoch']}_"
        f"{dataset_name}.torch"
    )

    if add_noise_to_load_data:
        model_name = model_name.replace(".torch", "_noise.torch")

    # ------------------------------------------------------------------
    # TensorBoard run
    # ------------------------------------------------------------------
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    tensorboard_root = (
        Path(args.tb_root).expanduser().resolve()
        if args.tb_root is not None
        else repo_root / "runs"
    )
    tensorboard_log_dir = (
        tensorboard_root
        / f"{Path(model_name).stem}_{run_stamp}"
    )
    tensorboard_log_dir.mkdir(parents=True, exist_ok=False)

    writer = SummaryWriter(
        log_dir=str(tensorboard_log_dir),
        flush_secs=30,
    )

    writer.add_text("Config/model_name", model_name, 0)
    writer.add_text("Config/dataset", dataset_name, 0)
    writer.add_text("Config/train_labels", repr(train_labels), 0)
    writer.add_text("Config/test_labels", repr(test_labels), 0)
    writer.add_text("Config/hyperparameters", repr(hyper), 0)
    writer.add_text(
        "Config/LSTM_input",
        (
            f"history_length={hist_length}; "
            f"features={n_lstm_input}; "
            "input=q_history+qdot_history; torque_history=excluded"
        ),
        0,
    )

    writer.add_custom_scalars(
        {
            "Loss comparison": {
                "Normalized torque MSE": [
                    "Multiline",
                    [
                        "Loss/train_torque_normalized",
                        "Loss/BT24_torque_normalized",
                    ],
                ],
            },
            "BT24 joints": {
                "Normalized joint MSE": [
                    "Multiline",
                    [
                        "BT24/normalized_hip_mse",
                        "BT24/normalized_knee_mse",
                    ],
                ],
            },
        }
    )

    print("\n################################################")
    print("TensorBoard:")
    print(f"Log directory: {tensorboard_log_dir}")
    print(
        "Start TensorBoard with:\n"
        f'tensorboard --logdir "{tensorboard_root}" --port 6006'
    )

    # ------------------------------------------------------------------
    # Construct or load model
    # ------------------------------------------------------------------
    if load_model:
        load_file = (
            learning_dir
            / "trained_models"
            / model_type_folder
            / model_name
        )
        print(f"Loading model: {load_file}")
        state = torch.load(
            load_file,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        delan_model = nn_type(n_dof, **state["hyper"])
        delan_model.load_state_dict(state["state_dict"])
    else:
        delan_model = nn_type(n_dof, **hyper)

    delan_model = delan_model.cuda() if cuda else delan_model.cpu()

    optimizer = torch.optim.Adam(
        delan_model.parameters(),
        lr=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
        amsgrad=False,
    )

    # ------------------------------------------------------------------
    # Replay memory
    # ------------------------------------------------------------------
    if not full_model:
        lstm_input_shape = (
            (hist_length, n_lstm_input)
            if hist_length > 0
            else (1,)
        )
    else:
        train_lstm_input = train_lstm_input.reshape(
            train_qp.shape[0],
            -1,
        )
        lstm_input_shape = (train_lstm_input.shape[-1],)

    mem_dim = (
        (n_dof,),
        (n_dof,),
        (n_dof,),
        (n_dof,),
        lstm_input_shape,
    )
    mem = PyTorchReplayMemory(
        train_qp.shape[0],
        hyper["n_minibatch"],
        mem_dim,
        cuda,
    )
    mem.add_samples(
        [
            train_qp,
            train_qv,
            train_qa,
            train_tau,
            train_lstm_input,
        ]
    )

    # ------------------------------------------------------------------
    # Loss normalization
    # ------------------------------------------------------------------
    if flag_normalize_tau:
        norm_tau = torch.from_numpy(
            np.var(train_tau, axis=0)
        )
    else:
        norm_tau = torch.ones(n_dof)

    norm_tau = (
        norm_tau.cuda()
        if cuda
        else norm_tau.cpu()
    )
    norm_tau_np = norm_tau.detach().cpu().numpy()

    if torch.any(norm_tau <= 0):
        raise RuntimeError(
            f"Torque variance must be positive, got {norm_tau_np}."
        )

    # ------------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------------
    init_param_dict = delan_model.state_dict()
    total_parameters = 0
    lstm_parameters = 0

    for key, value in init_param_dict.items():
        parameter_count = int(value.reshape(-1).shape[0])
        total_parameters += parameter_count
        if "lstm" in key:
            lstm_parameters += parameter_count

    print(
        f"Number of parameters {total_parameters} | "
        f"LSTM {lstm_parameters} | "
        f"MLPs {total_parameters - lstm_parameters}"
    )

    writer.add_scalar("Model/total_parameters", total_parameters, 0)
    writer.add_scalar("Model/LSTM_parameters", lstm_parameters, 0)
    writer.add_scalar(
        "Model/non_LSTM_parameters",
        total_parameters - lstm_parameters,
        0,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epoch_i = 0
    t0_start = time.perf_counter()
    accumulated_console_time = 0.0

    try:
        while epoch_i < hyper["max_epoch"] and not load_model:
            epoch_loss = 0.0
            epoch_inv_dyn_mean = 0.0
            epoch_inv_dyn_var = 0.0
            epoch_power_mean = 0.0
            epoch_power_var = 0.0
            epoch_joint_loss = np.zeros(n_dof, dtype=np.float64)
            n_batches = 0

            epoch_start = time.perf_counter()

            if (
                save_checkpoint_model
                and epoch_i > 0
                and epoch_i % checkpoint_period == 0
            ):
                checkpoint_dir = (
                    learning_dir
                    / "trained_models"
                    / model_type_folder
                    / "checkpoint"
                )
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = (
                    checkpoint_dir
                    / f"{epoch_i}_{model_name}"
                )

                print(f"Saving checkpoint model epoch: {epoch_i}")
                torch.save(
                    {
                        "epoch": epoch_i,
                        "hyper": hyper,
                        "state_dict": delan_model.state_dict(),
                    },
                    checkpoint_path,
                )

            delan_model.train()

            for q, qd, qdd, tau, lstm_input in mem:
                optimizer.zero_grad()

                if hist_length == 0:
                    tau_hat, dEdt_hat = delan_model(q, qd, qdd)
                else:
                    tau_hat, dEdt_hat = delan_model(
                        q,
                        qd,
                        qdd,
                        lstm_input,
                    )

                normalized_squared_error = (
                    (tau_hat - tau) ** 2 / norm_tau
                )
                err_inv = torch.sum(
                    normalized_squared_error,
                    dim=1,
                )
                loss_inv_dyn = torch.mean(err_inv)
                var_inv_dyn = torch.var(err_inv)

                dEdt = torch.matmul(
                    qd.view(-1, n_dof, 1).transpose(
                        dim0=1,
                        dim1=2,
                    ),
                    tau.view(-1, n_dof, 1),
                ).view(-1)
                err_dEdt = (dEdt_hat - dEdt) ** 2
                loss_power_conservation = torch.mean(err_dEdt)
                var_power_conservation = torch.var(err_dEdt)

                if loss_power:
                    loss = (
                        loss_inv_dyn
                        + loss_power_conservation
                    )
                else:
                    loss = loss_inv_dyn

                loss.backward()
                optimizer.step()

                n_batches += 1
                epoch_loss += float(loss.item())
                epoch_inv_dyn_mean += float(loss_inv_dyn.item())
                epoch_inv_dyn_var += float(var_inv_dyn.item())
                epoch_power_mean += float(
                    loss_power_conservation.item()
                )
                epoch_power_var += float(
                    var_power_conservation.item()
                )
                epoch_joint_loss += (
                    normalized_squared_error
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                )

            if n_batches == 0:
                raise RuntimeError("Replay memory produced no batches.")

            epoch_loss /= float(n_batches)
            epoch_inv_dyn_mean /= float(n_batches)
            epoch_inv_dyn_var /= float(n_batches)
            epoch_power_mean /= float(n_batches)
            epoch_power_var /= float(n_batches)
            epoch_joint_loss /= float(n_batches)

            epoch_i += 1
            epoch_time = time.perf_counter() - epoch_start
            accumulated_console_time += epoch_time

            # TensorBoard training scalars: written after every epoch.
            writer.add_scalar(
                "Loss/train_total",
                epoch_loss,
                epoch_i,
            )
            writer.add_scalar(
                "Loss/train_torque_normalized",
                epoch_inv_dyn_mean,
                epoch_i,
            )
            writer.add_scalar(
                "Loss/train_power",
                epoch_power_mean,
                epoch_i,
            )
            writer.add_scalar(
                "Train/normalized_hip_mse",
                float(epoch_joint_loss[0]),
                epoch_i,
            )
            writer.add_scalar(
                "Train/normalized_knee_mse",
                float(epoch_joint_loss[1]),
                epoch_i,
            )
            writer.add_scalar(
                "Optimization/learning_rate",
                optimizer.param_groups[0]["lr"],
                epoch_i,
            )
            writer.add_scalar(
                "Timing/seconds_per_epoch",
                epoch_time,
                epoch_i,
            )

            # BT24 validation: written periodically to limit evaluation overhead.
            if (
                epoch_i == 1
                or epoch_i % args.tb_eval_period == 0
                or epoch_i == hyper["max_epoch"]
            ):
                bt24_metrics = evaluate_torque_model(
                    model=delan_model,
                    qp=test_qp,
                    qv=test_qv,
                    qa=test_qa,
                    tau=test_tau,
                    lstm_history=test_lstm_input,
                    norm_tau=norm_tau,
                    hist_length=hist_length,
                    batch_size=args.tb_eval_batch_size,
                )

                writer.add_scalar(
                    "Loss/BT24_torque_normalized",
                    bt24_metrics["normalized_total_mse"],
                    epoch_i,
                )
                writer.add_scalar(
                    "Loss/BT24_torque_raw",
                    bt24_metrics["raw_total_mse"],
                    epoch_i,
                )
                writer.add_scalar(
                    "BT24/normalized_hip_mse",
                    float(
                        bt24_metrics[
                            "normalized_joint_mse"
                        ][0]
                    ),
                    epoch_i,
                )
                writer.add_scalar(
                    "BT24/normalized_knee_mse",
                    float(
                        bt24_metrics[
                            "normalized_joint_mse"
                        ][1]
                    ),
                    epoch_i,
                )
                writer.add_scalar(
                    "BT24/raw_hip_mse",
                    float(
                        bt24_metrics["raw_joint_mse"][0]
                    ),
                    epoch_i,
                )
                writer.add_scalar(
                    "BT24/raw_knee_mse",
                    float(
                        bt24_metrics["raw_joint_mse"][1]
                    ),
                    epoch_i,
                )

                context_values = bt24_metrics["context_values"]
                if context_values is not None:
                    writer.add_scalar(
                        "Context/BT24_z_mean",
                        float(context_values.mean().item()),
                        epoch_i,
                    )
                    writer.add_scalar(
                        "Context/BT24_z_std",
                        float(context_values.std().item()),
                        epoch_i,
                    )
                    writer.add_histogram(
                        "Context/BT24_z_distribution",
                        context_values,
                        epoch_i,
                    )

                print(
                    f"BT24 epoch {epoch_i:05d}: "
                    f"normalized torque MSE = "
                    f"{bt24_metrics['normalized_total_mse']:.6e}"
                )

            # Parameter histograms reveal saturation or collapsed weights.
            if (
                epoch_i == 1
                or epoch_i % args.tb_histogram_period == 0
                or epoch_i == hyper["max_epoch"]
            ):
                for parameter_name, parameter in (
                    delan_model.named_parameters()
                ):
                    writer.add_histogram(
                        f"Parameters/{parameter_name}",
                        parameter.detach().cpu(),
                        epoch_i,
                    )

            if (
                epoch_i == 1
                or epoch_i % console_log_period == 0
            ):
                average_console_epoch_time = (
                    accumulated_console_time
                    / (
                        1
                        if epoch_i == 1
                        else console_log_period
                    )
                )

                print(
                    f"Epoch {epoch_i:05d}: "
                    f"Time = "
                    f"{time.perf_counter() - t0_start:05.1f}s, "
                    f"Time/Epoch = "
                    f"{average_console_epoch_time:05.4f}s, "
                    f"Loss = {epoch_loss:.3e}, "
                    f"Inv Dyn = {epoch_inv_dyn_mean:.3e} "
                    f"± {1.96 * np.sqrt(epoch_inv_dyn_var):.3e}, "
                    f"Power Con = {epoch_power_mean:.3e} "
                    f"± {1.96 * np.sqrt(epoch_power_var):.3e}"
                )
                accumulated_console_time = 0.0

            writer.flush()

        # --------------------------------------------------------------
        # Save final model
        # --------------------------------------------------------------
        if save_model and not load_model:
            model_directory = (
                learning_dir
                / "trained_models"
                / model_type_folder
            )
            model_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            model_path = model_directory / model_name

            torch.save(
                {
                    "epoch": epoch_i,
                    "hyper": hyper,
                    "state_dict": delan_model.state_dict(),
                },
                model_path,
            )
            print(f"Saved final model: {model_path}")
            writer.add_text(
                "Output/model_path",
                str(model_path),
                epoch_i,
            )

        # --------------------------------------------------------------
        # Final DeLaN evaluation and torque plots
        # --------------------------------------------------------------
        print("\n################################################")
        print("Evaluating DeLaN:")

        evaluation_start = time.perf_counter()

        q = torch.from_numpy(test_qp).float().to(
            delan_model.device
        )
        qd = torch.from_numpy(test_qv).float().to(
            delan_model.device
        )
        qdd = torch.from_numpy(test_qa).float().to(
            delan_model.device
        )
        lstm_input = torch.from_numpy(
            test_lstm_input
        ).float().to(delan_model.device)
        zeros = torch.zeros_like(q).float().to(
            delan_model.device
        )

        delan_model.eval()
        with torch.no_grad():
            if hist_length == 0:
                delan_g = (
                    delan_model.inv_dyn(
                        q,
                        zeros,
                        zeros,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                delan_c = (
                    delan_model.inv_dyn(
                        q,
                        qd,
                        zeros,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                    - delan_g
                )
                delan_m = (
                    delan_model.inv_dyn(
                        q,
                        zeros,
                        qdd,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                    - delan_g
                )
                delan_output = delan_model(q, qd, qdd)
            else:
                delan_g = (
                    delan_model.inv_dyn(
                        q,
                        zeros,
                        zeros,
                        lstm_input,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                delan_c = (
                    delan_model.inv_dyn(
                        q,
                        qd,
                        zeros,
                        lstm_input,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                    - delan_g
                )
                delan_m = (
                    delan_model.inv_dyn(
                        q,
                        zeros,
                        qdd,
                        lstm_input,
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                    - delan_g
                )
                delan_output = delan_model(
                    q,
                    qd,
                    qdd,
                    lstm_input,
                )

            delan_tau = delan_output[0].cpu().numpy()
            delan_dEdt = delan_output[1].cpu().numpy()

        computation_time = (
            time.perf_counter() - evaluation_start
        ) / (3.0 * float(test_qp.shape[0]))

        test_dEdt = np.sum(test_tau * test_qv, axis=1)
        err_g = (
            np.sum((delan_g - test_g) ** 2 / norm_tau_np)
            / float(test_qp.shape[0])
        )
        err_m = (
            np.sum((delan_m - test_m) ** 2 / norm_tau_np)
            / float(test_qp.shape[0])
        )
        err_c = (
            np.sum((delan_c - test_c) ** 2 / norm_tau_np)
            / float(test_qp.shape[0])
        )
        err_tau = (
            np.sum((delan_tau - test_tau) ** 2 / norm_tau_np)
            / float(test_qp.shape[0])
        )
        err_dEdt = (
            np.sum((delan_dEdt - test_dEdt) ** 2)
            / float(test_qp.shape[0])
        )

        print("\nPerformance:")
        print(f"                Torque MSE = {err_tau:.3e}")
        print(f"              Inertial MSE = {err_m:.3e}")
        print(
            "Coriolis & Centrifugal MSE = "
            f"{err_c:.3e}"
        )
        print(f"         Gravitational MSE = {err_g:.3e}")
        print(
            f"    Power Conservation MSE = {err_dEdt:.3e}"
        )
        print(
            "      Comp Time per Sample = "
            f"{computation_time:.3e}s / "
            f"{1.0 / computation_time:.1f}Hz"
        )

        writer.add_scalar(
            "Final/torque_mse_normalized",
            float(err_tau),
            epoch_i,
        )
        writer.add_scalar(
            "Final/inertial_mse_normalized",
            float(err_m),
            epoch_i,
        )
        writer.add_scalar(
            "Final/coriolis_mse_normalized",
            float(err_c),
            epoch_i,
        )
        writer.add_scalar(
            "Final/gravity_mse_normalized",
            float(err_g),
            epoch_i,
        )
        writer.add_scalar(
            "Final/power_mse",
            float(err_dEdt),
            epoch_i,
        )

        figure_directory = (
            learning_dir
            / "figures"
            / "mpc_DeLaN_Performance"
            / model_type_folder
            / model_name
        )
        plot_torques(
            test_tau,
            test_m,
            test_c,
            test_g,
            delan_tau,
            delan_m,
            delan_c,
            delan_g,
            test_labels,
            divider,
            str(figure_directory),
            render,
        )

        writer.add_text(
            "Output/figure_directory",
            str(figure_directory),
            epoch_i,
        )

    finally:
        writer.flush()
        writer.close()
        print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
