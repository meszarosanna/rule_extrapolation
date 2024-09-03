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


def sweep2df(sweep_runs, filename, save=False, load=False, pick_max=True):
    if pick_max is True:
        csv_name = f"{filename}.csv"
        npy_name = f"{filename}.npz"
    else:
        csv_name = f"{filename}_at_min_val_loss.csv"
        npy_name = f"{filename}_at_min_val_loss.npz"

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

        rule_1_histories = npy_data["rule_1_history"]
        try:
            rule_3_histories = npy_data["rule_3_history"]
        except:
            rule_3_histories = []
        rule_2_histories = npy_data["rule_2_history"]

        grammatical_histories = npy_data["grammatical_history"]

        ood_rule_1_histories = npy_data["ood_rule_1_history"]
        try:
            ood_rule_3_histories = npy_data["ood_rule_3_history"]
        except:
            ood_rule_3_histories = []
        ood_rule_2_histories = npy_data["ood_rule_2_history"]
        ood_rule_2_completion_histories = npy_data["ood_rule_2_completion_history"]
        ood_grammatical_histories = npy_data["ood_grammatical_history"]

        sos_rule_1_histories = npy_data["sos_rule_1_history"]
        try:
            sos_rule_3_histories = npy_data["sos_rule_3_history"]
        except:
            sos_rule_3_histories = []
        sos_rule_2_histories = npy_data["sos_rule_2_history"]
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
            rule_1_histories,
            rule_3_histories,
            rule_2_histories,
            grammatical_histories,
            ood_rule_1_histories,
            ood_rule_3_histories,
            ood_rule_2_completion_histories,
            ood_rule_2_histories,
            ood_grammatical_histories,
            sos_rule_1_histories,
            sos_rule_3_histories,
            sos_rule_2_histories,
            sos_grammatical_histories,
        )

    data = []
    train_loss_histories = []
    val_loss_histories = []
    val_kl_histories = []
    val_accuracy_histories = []

    rule_1_histories = []
    rule_3_histories = []
    rule_2_histories = []
    finised_histories = []
    grammatical_histories = []

    ood_rule_1_histories = []
    ood_rule_3_histories = []
    ood_rule_2_completion_histories = []
    ood_rule_2_histories = []
    ood_finised_histories = []
    ood_grammatical_histories = []

    sos_rule_1_histories = []
    sos_rule_3_histories = []
    sos_rule_2_histories = []
    sos_finised_histories = []
    sos_grammatical_histories = []

    for run in sweep_runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict
        try:
            summary["epoch"]
        except:
            continue

        if run.state == "finished" or summary["epoch"] > 100:
            print(f"\t Processing {run.name}...")
            # try:
            if True:
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config = {k: v for k, v in run.config.items() if not k.startswith("_")}

                # general
                test_prompt_length = config["model.test_prompt_length"]
                max_pred_length = config["model.max_pred_length"]
                lr = config["model.lr"]
                max_length = config["data.max_length"]
                batch_size = config["data.batch_size"]
                seed_everything = config["seed_everything"]
                model = config["model.model"]

                try:
                    optimizer = config["model.optimizer"]
                except:
                    optimizer = config["optimizer"]

                try:
                    adversarial_training = config["model.adversarial_training"]
                except:
                    adversarial_training = False

                # transformer
                dim_model = config[
                    "model.dim_model"
                ]  # also embedding dim in LSTM, linear
                num_heads = config["model.num_heads"]
                dim_feedforward = config["model.dim_feedforward"]
                num_decoder_layers = config["model.num_decoder_layers"]
                dropout_p = config["model.dropout_p"]
                layer_norm_eps = config["model.layer_norm_eps"]

                # lstm
                lstm_embedding_dim = config["model.dim_model"]
                try:
                    lstm_hidden_dim = config["model.hidden_dim"]
                    lstm_num_layers = config["model.num_layers"]
                    lstm_dropout = config["model.dropout"]
                except:
                    lstm_hidden_dim = config["hidden_dim"]
                    lstm_num_layers = config["num_layers"]
                    lstm_dropout = config["dropout"]

                # xlstm
                try:
                    xlstm_embedding_dim = config["model.xlstm_embedding_dim"]
                    xlstm_num_blocks = config["model.num_blocks"]
                except:
                    xlstm_embedding_dim = 0
                    xlstm_num_blocks = 0

                # linear
                linear_embedding_dim = config["model.dim_model"]
                try:
                    linear_bias = config["model.bias"]
                except:
                    linear_bias = config["bias"]
                linear_dim = config["data.max_length"]

                # mamba
                try:
                    mamba_d_model = config["model.d_model"]
                    mamba_d_state = config["model.d_state"]
                    mamba_d_conv = config["model.d_conv"]
                    mamba_n_layers = config["model.n_layers"]
                except:
                    mamba_d_model = config["d_model"]
                    mamba_d_state = config["d_state"]
                    mamba_d_conv = config["d_conv"]
                    mamba_n_layers = config["n_layers"]

                # training stats
                train_loss_history = run.history(keys=[f"Train/loss"])
                train_loss_histories.append(train_loss_history["Train/loss"])

                try:
                    min_train_loss_step, min_train_loss = (
                        train_loss_history.idxmin()[1],
                        train_loss_history.min()[1],
                    )
                except:
                    print(f"\t\t Skipping {run.name} due to NaN train loss")
                    train_loss_histories.pop()
                    continue

                # validation stats
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

                pick_metric = (
                    lambda history: history.max()[1]
                    if pick_max
                    else history.iloc[int(min_val_loss_step)][key]
                )

                # ID
                key = f"Val/ID/finished_accuracy"
                history = run.history(keys=[key])
                finished4min_val_loss = pick_metric(history)
                finised_histories.append(history[key])

                key = f"Val/ID/finished/rule_1_accuracy"
                history = run.history(keys=[key])
                rule_1_accuracy4min_val_loss = pick_metric(history)
                rule_1_histories.append(history[key])

                try:
                    key = f"Val/ID/finished/rule_3_accuracy"
                    history = run.history(keys=[key])
                    rule_3_accuracy4min_val_loss = pick_metric(history)
                    rule_3_histories.append(history[key])
                except:
                    rule_3_accuracy4min_val_loss = 0
                    rule_3_histories.append([])

                key = f"Val/ID/finished/rule_2_accuracy"
                history = run.history(keys=[key])
                rule_2_accuracy4min_val_loss = pick_metric(history)
                rule_2_histories.append(history[key])

                key = f"Val/ID/finished/grammatical_accuracy"
                history = run.history(keys=[key])
                grammatical_accuracy4min_val_loss = pick_metric(history)
                grammatical_histories.append(history[key])

                # OOD
                key = f"Val/OOD/finished_accuracy"
                history = run.history(keys=[key])
                ood_finished4min_val_loss = pick_metric(history)
                ood_finised_histories.append(history[key])

                key = f"Val/OOD/finished/rule_1_accuracy"
                history = run.history(keys=[key])
                ood_rule_1_accuracy4min_val_loss = pick_metric(history)
                ood_rule_1_histories.append(history[key])

                try:
                    key = f"Val/OOD/finished/rule_3_accuracy"
                    history = run.history(keys=[key])
                    ood_rule_3_accuracy4min_val_loss = pick_metric(history)
                    ood_rule_3_histories.append(history[key])
                except:
                    ood_rule_3_accuracy4min_val_loss = 0
                    ood_rule_3_histories.append([])

                key = f"Val/OOD/finished/rule_2_accuracy"
                history = run.history(keys=[key])
                ood_rule_2_accuracy4min_val_loss = pick_metric(history)
                ood_rule_2_histories.append(history[key])

                key = f"Val/OOD/finished/rule_2_completion_accuracy"
                history = run.history(keys=[key])
                ood_rule_2_completion_accuracy4min_val_loss = pick_metric(history)
                ood_rule_2_completion_histories.append(history[key])

                key = f"Val/OOD/finished/grammatical_accuracy"
                history = run.history(keys=[key])
                ood_grammatical_accuracy4min_val_loss = pick_metric(history)
                ood_grammatical_histories.append(history[key])

                # SOS

                key = f"Val/SOS/finished_accuracy"
                history = run.history(keys=[key])
                sos_finished4min_val_loss = pick_metric(history)
                sos_finised_histories.append(history[key])

                key = f"Val/SOS/finished/rule_1_accuracy"
                history = run.history(keys=[key])
                sos_rule_1_accuracy4min_val_loss = pick_metric(history)
                sos_rule_1_histories.append(history[key])

                try:
                    key = f"Val/SOS/finished/rule_3_accuracy"
                    history = run.history(keys=[key])
                    sos_rule_3_accuracy4min_val_loss = pick_metric(history)
                    sos_rule_3_histories.append(history[key])
                except:
                    sos_rule_3_accuracy4min_val_loss = 0
                    sos_rule_3_histories.append([])

                key = f"Val/SOS/finished/rule_2_accuracy"
                history = run.history(keys=[key])
                sos_rule_2_accuracy4min_val_loss = pick_metric(history)
                sos_rule_2_histories.append(history[key])

                key = f"Val/SOS/finished/grammatical_accuracy"
                history = run.history(keys=[key])
                sos_grammatical_accuracy4min_val_loss = pick_metric(history)
                sos_grammatical_histories.append(history[key])

                data.append(
                    [
                        run.name,
                        seed_everything,
                        test_prompt_length,
                        max_pred_length,
                        lr,
                        adversarial_training,
                        max_length,
                        batch_size,
                        model,
                        optimizer,
                        # transformer
                        dim_model,
                        num_heads,
                        dim_feedforward,
                        num_decoder_layers,
                        dropout_p,
                        layer_norm_eps,
                        # lstm
                        lstm_embedding_dim,
                        lstm_hidden_dim,
                        lstm_num_layers,
                        lstm_dropout,
                        # linear
                        linear_embedding_dim,
                        linear_bias,
                        linear_dim,
                        # mamba
                        mamba_d_model,
                        mamba_d_state,
                        mamba_d_conv,
                        mamba_n_layers,
                        # xlstm
                        xlstm_embedding_dim,
                        xlstm_num_blocks,
                        # train stats
                        min_train_loss,
                        min_train_loss_step,
                        # val stats
                        max_val_accuracy,
                        max_val_accuracy_step,
                        min_val_loss,
                        min_val_loss_step,
                        min_val_kl,
                        min_val_kl_step,
                        # ID
                        rule_1_accuracy4min_val_loss,
                        rule_3_accuracy4min_val_loss,
                        rule_2_accuracy4min_val_loss,
                        grammatical_accuracy4min_val_loss,
                        finished4min_val_loss,
                        # OOD
                        ood_rule_1_accuracy4min_val_loss,
                        ood_rule_3_accuracy4min_val_loss,
                        ood_rule_2_completion_accuracy4min_val_loss,
                        ood_rule_2_accuracy4min_val_loss,
                        ood_grammatical_accuracy4min_val_loss,
                        ood_finished4min_val_loss,
                        # SOS
                        sos_rule_1_accuracy4min_val_loss,
                        sos_rule_3_accuracy4min_val_loss,
                        sos_rule_2_accuracy4min_val_loss,
                        sos_grammatical_accuracy4min_val_loss,
                        sos_finished4min_val_loss,
                    ]
                )

            # except:
            #     print(f"Encountered a faulty run with ID {run.name}")

    runs_df = pd.DataFrame(
        data,
        columns=[
            "name",
            "seed_everything",
            "test_prompt_length",
            "max_pred_length",
            "lr",
            "adversarial_training",
            "max_length",
            "batch_size",
            "model",
            "optimizer",
            # transformer
            "dim_model",
            "num_heads",
            "dim_feedforward",
            "num_decoder_layers",
            "dropout_p",
            "layer_norm_eps",
            # lstm
            "lstm_embedding_dim",
            "lstm_hidden_dim",
            "lstm_num_layers",
            "lstm_dropout",
            # xlstm
            "xlstm_embedding_dim",
            "xlstm_num_blocks",
            # linear
            "linear_embedding_dim",
            "linear_bias",
            "linear_dim",
            # mamba
            "mamba_d_model",
            "mamba_d_state",
            "mamba_d_conv",
            "mamba_n_layers",
            # train stats
            "min_train_loss",
            "min_train_loss_step",
            # val stats
            "max_val_accuracy",
            "max_val_accuracy_step",
            "min_val_loss",
            "min_val_loss_step",
            "min_val_kl",
            "min_val_kl_step",
            # ID
            "rule_1_accuracy4min_val_loss",
            "rule_3_accuracy4min_val_loss",
            "rule_2_accuracy4min_val_loss",
            "grammatical_accuracy4min_val_loss",
            "finished4min_val_loss",
            # OOD
            "ood_rule_1_accuracy4min_val_loss",
            "ood_rule_3_accuracy4min_val_loss",
            "ood_rule_2_completion_accuracy4min_val_loss",
            "ood_rule_2_accuracy4min_val_loss",
            "ood_grammatical_accuracy4min_val_loss",
            "ood_finished4min_val_loss",
            # SOS
            "sos_rule_1_accuracy4min_val_loss",
            "sos_rule_3_accuracy4min_val_loss",
            "sos_rule_2_accuracy4min_val_loss",
            "sos_grammatical_accuracy4min_val_loss",
            "sos_finished4min_val_loss",
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
            rule_1_history=_prune_histories(rule_1_histories),
            rule_3_history=_prune_histories(rule_3_histories),
            rule_2_history=_prune_histories(rule_2_histories),
            grammatical_history=_prune_histories(grammatical_histories),
            ood_rule_1_history=_prune_histories(ood_rule_1_histories),
            ood_rule_3_history=_prune_histories(ood_rule_3_histories),
            ood_rule_2_completion_history=_prune_histories(
                ood_rule_2_completion_histories
            ),
            ood_rule_2_history=_prune_histories(ood_rule_2_histories),
            ood_grammatical_history=_prune_histories(ood_grammatical_histories),
            sos_rule_1_history=_prune_histories(sos_rule_1_histories),
            sos_rule_3_history=_prune_histories(sos_rule_3_histories),
            sos_rule_2_history=_prune_histories(sos_rule_2_histories),
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
        rule_1_histories,
        rule_3_histories,
        rule_2_histories,
        grammatical_histories,
        ood_rule_1_histories,
        ood_rule_3_histories,
        ood_rule_2_completion_histories,
        ood_rule_2_histories,
        ood_grammatical_histories,
        sos_rule_1_histories,
        sos_rule_3_histories,
        sos_rule_2_histories,
        sos_grammatical_histories,
    )


def stats2string(df):
    s = [
        f"${m:.3f}\scriptscriptstyle\pm {s:.3f}$ & "
        for m, s in zip(df.mean().train_mcc, df.std().train_mcc)
    ]
    return "".join(s)


def rule_stats2string_per_model(
    stats,
    plot=("val_loss", "rule_1", "rule_2", "ood_rule_1", "ood_rule_2_completion"),
    include_r2=True,
):
    model_colors = {
        "transformer": "figblue",
        "lstm": "orange",
        "linear": "green!80!black",
        "mamba": "figred",
        "xlstm": "purple",
    }
    models = sorted(stats["rule_1"].groups.keys())
    print("------------------------------")
    print(f"Model order is={models}")
    print(f"Plot order is={plot}")
    print("------------------------------")
    table = []
    for model in models:
        row = []
        for p in plot:
            if include_r2 is False and p == "rule_2":
                continue

            stat = stats[p].get_group(model)
            row.append(f"${stat.mean():.3f}\scriptscriptstyle\pm {stat.std():.3f}$ & ")
        # convert model name to have a capital starting letter

        # strip last & symbol
        row[-1] = row[-1][:-2]

        print(
            r"{\color{"
            + model_colors[model]
            + "}"
            + (model.capitalize() if model != "lstm" else model.upper())
            + "}"
            + " &"
            + "".join(row)
            + r"\\"
        )
    return table


def grouped_rule_stats(df, groupby_key="model"):
    grouped_df = df.groupby(groupby_key)

    stats = {}

    stats["val_loss"] = grouped_df.min_val_loss

    stats["rule_1"] = grouped_df.rule_1_accuracy4min_val_loss
    stats["rule_2"] = grouped_df.rule_2_accuracy4min_val_loss
    try:
        stats["rule_3"] = grouped_df.rule_3_accuracy4min_val_loss
    except:
        pass
    stats["grammatical"] = grouped_df.grammatical_accuracy4min_val_loss
    stats["finished"] = grouped_df.finished4min_val_loss

    stats["ood_rule_1"] = grouped_df.ood_rule_1_accuracy4min_val_loss
    try:
        stats["ood_rule_3"] = grouped_df.ood_rule_3_accuracy4min_val_loss
    except:
        pass
    stats["ood_rule_2"] = grouped_df.ood_rule_2_accuracy4min_val_loss
    stats[
        "ood_rule_2_completion"
    ] = grouped_df.ood_rule_2_completion_accuracy4min_val_loss
    stats["ood_grammatical"] = grouped_df.ood_grammatical_accuracy4min_val_loss
    stats["ood_finished"] = grouped_df.ood_finished4min_val_loss

    stats["sos_rule_1"] = grouped_df.sos_rule_1_accuracy4min_val_loss
    stats["sos_rule_2"] = grouped_df.sos_rule_2_accuracy4min_val_loss
    stats["sos_grammatical"] = grouped_df.sos_grammatical_accuracy4min_val_loss
    stats["sos_finished"] = grouped_df.sos_finished4min_val_loss

    return stats
