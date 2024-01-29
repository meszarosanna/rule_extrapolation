from os.path import isfile

import numpy as np
import pandas as pd

BLUE = "#1A85FF"
RED = "#D0021B"
METRIC_EPS = 1e-6

from matplotlib import rc


def plot_typography(
    usetex: bool = False, small: int = 16, medium: int = 20, big: int = 22
):
    """
    Initializes font settings and visualization backend (LaTeX or standard matplotlib).
    :param usetex: flag to indicate the usage of LaTeX (needs LaTeX indstalled)
    :param small: small font size in pt (for legends and axes' ticks)
    :param medium: medium font size in pt (for axes' labels)
    :param big: big font size in pt (for titles)
    :return:
    """

    # font family
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})

    # backend
    rc("text", usetex=usetex)
    rc("font", family="serif")

    # font sizes
    rc("font", size=small)  # controls default text sizes
    rc("axes", titlesize=big)  # fontsize of the axes title
    rc("axes", labelsize=medium)  # fontsize of the x and y labels
    rc("xtick", labelsize=small)  # fontsize of the tick labels
    rc("ytick", labelsize=small)  # fontsize of the tick labels
    rc("legend", fontsize=small)  # legend fontsize
    rc("figure", titlesize=big)  # fontsize of the figure title


def sweep2df(
    sweep_runs,
    filename,
    save=False,
    load=False,
):
    csv_name = f"{filename}.csv"
    npy_name = f"{filename}.npz"

    if load is True and isfile(csv_name) is True and isfile(npy_name) is True:
        print(f"\t Loading {filename}...")
        npy_data = np.load(npy_name)

        train_loss_histories = npy_data["train_loss_history"]
        val_loss_histories = npy_data["val_loss_history"]
        val_kl_histories = npy_data["val_kl_history"]
        val_accuracy_histories = npy_data["val_accuracy_history"]

        finised_histories = npy_data["finised_history"]
        ood_finised_histories = npy_data["ood_finised_history"]
        sos_finised_histories = npy_data["sos_finised_history"]

        as_before_bs_histories = npy_data["as_before_bs_history"]
        same_as_bs_histories = npy_data["same_as_bs_history"]
        grammatical_histories = npy_data["grammatical_history"]

        ood_as_before_bs_histories = npy_data["ood_as_before_bs_history"]
        ood_as_before_bs_completion_histories = npy_data[
            "ood_as_before_bs_completion_history"
        ]
        ood_same_as_bs_histories = npy_data["ood_same_as_bs_history"]
        ood_grammatical_histories = npy_data["ood_grammatical_history"]

        sos_as_before_bs_histories = npy_data["sos_as_before_bs_history"]
        sos_same_as_bs_histories = npy_data["sos_same_as_bs_history"]
        sos_grammatical_histories = npy_data["sos_grammatical_history"]

        return (
            pd.read_csv(csv_name),
            train_loss_histories,
            val_loss_histories,
            val_kl_histories,
            val_accuracy_histories,
            finised_histories,
            ood_finised_histories,
            sos_finised_histories,
            as_before_bs_histories,
            same_as_bs_histories,
            grammatical_histories,
            ood_as_before_bs_histories,
            ood_as_before_bs_completion_histories,
            ood_same_as_bs_histories,
            ood_grammatical_histories,
            sos_as_before_bs_histories,
            sos_same_as_bs_histories,
            sos_grammatical_histories,
        )

    data = []
    train_loss_histories = []
    val_loss_histories = []
    val_kl_histories = []
    val_accuracy_histories = []

    as_before_bs_histories = []
    same_as_bs_histories = []
    finised_histories = []
    grammatical_histories = []

    ood_as_before_bs_histories = []
    ood_as_before_bs_completion_histories = []
    ood_same_as_bs_histories = []
    ood_finised_histories = []
    ood_grammatical_histories = []

    sos_as_before_bs_histories = []
    sos_same_as_bs_histories = []
    sos_finised_histories = []
    sos_grammatical_histories = []

    for run in sweep_runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        if run.state == "finished":
            # print(f"\t Processing {run.name}...")
            # try:
            if True:
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config = {k: v for k, v in run.config.items() if not k.startswith("_")}

                dim_model = config["model.dim_model"]
                num_heads = config["model.num_heads"]
                dim_feedforward = config["model.dim_feedforward"]
                num_decoder_layers = config["model.num_decoder_layers"]
                dropout_p = config["model.dropout_p"]
                test_prompt_length = config["model.test_prompt_length"]
                max_pred_length = config["model.max_pred_length"]
                lr = config["model.lr"]
                layer_norm_eps = config["model.layer_norm_eps"]

                try:
                    adversarial_training = config["model.adversarial_training"]
                except:
                    adversarial_training = False
                max_length = config["data.max_length"]
                batch_size = config["data.batch_size"]
                seed_everything = config["seed_everything"]

                train_loss_history = run.history(keys=[f"Train/loss"])
                train_loss_histories.append(train_loss_history["Train/loss"])

                min_train_loss_step, min_train_loss = (
                    train_loss_history.idxmin()[1],
                    train_loss_history.min()[1],
                )

                val_loss_history = run.history(keys=[f"Val/loss"])
                min_val_loss_step, min_val_loss = (
                    val_loss_history.idxmin()[1],
                    val_loss_history.min()[1],
                )

                val_loss_histories.append(val_loss_history["Val/loss"])

                val_kl_history = run.history(keys=[f"Val/kl"])
                val_kl_histories.append(val_kl_history["Val/kl"])
                min_val_kl_step, min_val_kl = (
                    val_kl_history.idxmin()[1],
                    val_kl_history.min()[1],
                )

                val_accuracy_history = run.history(keys=[f"Val/accuracy"])

                max_val_accuracy_step, max_val_accuracy = (
                    val_accuracy_history.idxmax()[1],
                    val_accuracy_history.max()[1],
                )

                val_accuracy_histories.append(val_accuracy_history["Val/accuracy"])

                finised_histories.append(
                    run.history(keys=[f"Val/ID/finished_accuracy"])[
                        f"Val/ID/finished_accuracy"
                    ]
                )
                ood_finised_histories.append(
                    run.history(keys=[f"Val/OOD/finished_accuracy"])[
                        f"Val/OOD/finished_accuracy"
                    ]
                )
                sos_finised_histories.append(
                    run.history(keys=[f"Val/SOS/finished_accuracy"])[
                        f"Val/SOS/finished_accuracy"
                    ]
                )

                key = f"Val/ID/finished/as_before_bs_accuracy"
                history = run.history(keys=[key])
                as_before_bs_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                as_before_bs_histories.append(history[key])

                key = f"Val/ID/finished/same_number_as_bs_accuracy"
                history = run.history(keys=[key])
                same_as_bs_accuracy4min_val_loss = history.iloc[int(min_val_loss_step)][
                    key
                ]
                same_as_bs_histories.append(history[key])

                key = f"Val/ID/finished/grammatical_accuracy"
                history = run.history(keys=[key])
                grammatical_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                grammatical_histories.append(history[key])

                key = f"Val/OOD/finished/as_before_bs_accuracy"
                history = run.history(keys=[key])
                ood_as_before_bs_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                ood_as_before_bs_histories.append(history[key])

                key = f"Val/OOD/finished/as_before_bs_completion_accuracy"
                history = run.history(keys=[key])
                ood_as_before_bs_completion_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                ood_as_before_bs_completion_histories.append(history[key])

                key = f"Val/OOD/finished/same_number_as_bs_accuracy"
                history = run.history(keys=[key])
                ood_same_as_bs_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                ood_same_as_bs_histories.append(history[key])

                key = f"Val/OOD/finished/grammatical_accuracy"
                history = run.history(keys=[key])
                ood_grammatical_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                ood_grammatical_histories.append(history[key])

                key = f"Val/SOS/finished/as_before_bs_accuracy"
                history = run.history(keys=[key])
                sos_as_before_bs_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                sos_as_before_bs_histories.append(history[key])

                key = f"Val/SOS/finished/same_number_as_bs_accuracy"
                history = run.history(keys=[key])
                sos_same_as_bs_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                sos_same_as_bs_histories.append(history[key])

                key = f"Val/SOS/finished/grammatical_accuracy"
                history = run.history(keys=[key])
                sos_grammatical_accuracy4min_val_loss = history.iloc[
                    int(min_val_loss_step)
                ][key]
                sos_grammatical_histories.append(history[key])

                data.append(
                    [
                        run.name,
                        seed_everything,
                        dim_model,
                        num_heads,
                        dim_feedforward,
                        num_decoder_layers,
                        dropout_p,
                        test_prompt_length,
                        max_pred_length,
                        lr,
                        layer_norm_eps,
                        adversarial_training,
                        max_length,
                        batch_size,
                        max_val_accuracy,
                        max_val_accuracy_step,
                        min_train_loss,
                        min_train_loss_step,
                        min_val_loss,
                        min_val_loss_step,
                        min_val_kl,
                        min_val_kl_step,
                        as_before_bs_accuracy4min_val_loss,
                        same_as_bs_accuracy4min_val_loss,
                        grammatical_accuracy4min_val_loss,
                        ood_as_before_bs_accuracy4min_val_loss,
                        ood_as_before_bs_completion_accuracy4min_val_loss,
                        ood_same_as_bs_accuracy4min_val_loss,
                        ood_grammatical_accuracy4min_val_loss,
                        sos_as_before_bs_accuracy4min_val_loss,
                        sos_same_as_bs_accuracy4min_val_loss,
                        sos_grammatical_accuracy4min_val_loss,
                    ]
                )

            # except:
            #     print(f"Encountered a faulty run with ID {run.name}")

    runs_df = pd.DataFrame(
        data,
        columns=[
            "name",
            "seed_everything",
            "dim_model",
            "num_heads",
            "dim_feedforward",
            "num_decoder_layers",
            "dropout_p",
            "test_prompt_length",
            "max_pred_length",
            "lr",
            "layer_norm_eps",
            "adversarial_training",
            "max_length",
            "batch_size",
            "max_val_accuracy",
            "max_val_accuracy_step",
            "min_train_loss",
            "min_train_loss_step",
            "min_val_loss",
            "min_val_loss_step",
            "min_val_kl",
            "min_val_kl_step",
            "as_before_bs_accuracy4min_val_loss",
            "same_as_bs_accuracy4min_val_loss",
            "grammatical_accuracy4min_val_loss",
            "ood_as_before_bs_accuracy4min_val_loss",
            "ood_as_before_bs_completion_accuracy4min_val_loss",
            "ood_same_as_bs_accuracy4min_val_loss",
            "ood_grammatical_accuracy4min_val_loss",
            "sos_as_before_bs_accuracy4min_val_loss",
            "sos_same_as_bs_accuracy4min_val_loss",
            "sos_grammatical_accuracy4min_val_loss",
        ],
    ).fillna(0)

    # Prune histories to the minimum length
    def _prune_histories(histories):
        min_len = np.array([len(v) for v in histories]).min()
        return np.array([v[:min_len] for v in histories])

    if save is True:
        runs_df.to_csv(csv_name)
        np.savez_compressed(
            npy_name,
            train_loss_history=_prune_histories(train_loss_histories),
            val_loss_history=_prune_histories(val_loss_histories),
            val_kl_history=_prune_histories(val_kl_histories),
            val_accuracy_history=_prune_histories(val_accuracy_histories),
            finised_history=_prune_histories(finised_histories),
            ood_finised_history=_prune_histories(ood_finised_histories),
            sos_finised_history=_prune_histories(sos_finised_histories),
            as_before_bs_history=_prune_histories(as_before_bs_histories),
            same_as_bs_history=_prune_histories(same_as_bs_histories),
            grammatical_history=_prune_histories(grammatical_histories),
            ood_as_before_bs_history=_prune_histories(ood_as_before_bs_histories),
            ood_as_before_bs_completion_history=_prune_histories(
                ood_as_before_bs_completion_histories
            ),
            ood_same_as_bs_history=_prune_histories(ood_same_as_bs_histories),
            ood_grammatical_history=_prune_histories(ood_grammatical_histories),
            sos_as_before_bs_history=_prune_histories(sos_as_before_bs_histories),
            sos_same_as_bs_history=_prune_histories(sos_same_as_bs_histories),
            sos_grammatical_history=_prune_histories(sos_grammatical_histories),
        )

    return (
        runs_df,
        train_loss_histories,
        val_loss_histories,
        val_kl_histories,
        val_accuracy_histories,
        finised_histories,
        ood_finised_histories,
        sos_finised_histories,
        as_before_bs_histories,
        same_as_bs_histories,
        grammatical_histories,
        ood_as_before_bs_histories,
        ood_as_before_bs_completion_histories,
        ood_same_as_bs_histories,
        ood_grammatical_histories,
        sos_as_before_bs_histories,
        sos_same_as_bs_histories,
        sos_grammatical_histories,
    )


def stats2string(df):
    s = [
        f"${m:.3f}\scriptscriptstyle\pm {s:.3f}$ & "
        for m, s in zip(df.mean().train_mcc, df.std().train_mcc)
    ]
    return "".join(s)
