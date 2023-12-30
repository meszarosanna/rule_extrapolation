import numpy as np

PAD = 4
SOS_token = np.array([2])
EOS_token = np.array([3])
BS = 64


def generate_aNbN_grammar_data(n, max_length=32):
    """
    PCFG with two rules:
    - number of a's and b's must be the same
    - a's come first, followed by b's

    :param n:
    :param max_length:
    :return:

    """

    lengths = np.random.randint(low=1, high=max_length // 2, size=n)

    data = []

    [
        data.append(
            np.concatenate((SOS_token, np.zeros(length), np.ones(length), EOS_token))
        )
        for length in lengths
    ]

    np.random.shuffle(data)

    return data


def generate_abN_grammar_data(n, max_length=32):
    """
    PCFG with one rule:
    - number of a's and b's must be the same

    :param n:
    :param max_length:
    :return:
    """

    lengths = np.random.randint(low=1, high=max_length // 2, size=n)

    data = []

    for lengths in lengths:
        abN = np.concatenate((np.zeros(lengths), np.ones(lengths)))
        # shuffle the symbols between start and end tokens
        np.random.shuffle(abN)
        data.append(np.concatenate((SOS_token, abN, EOS_token)))

    return data


def generate_aN_bM_grammar_data(n, max_length=32):
    """
    PCFG with one rule:
    - a's are before b's

    :param n:
    :param max_length:
    :return:
    """

    lengths_a = np.random.randint(low=1, high=max_length, size=n)
    lengths_b = np.ones_like(lengths_a) * max_length - lengths_a

    data = []

    for la, lb in zip(lengths_a, lengths_b):
        data.append(np.concatenate((SOS_token, np.zeros(la), np.ones(lb), EOS_token)))

    return data


def batchify_data(data, batch_size=BS, padding=True, padding_token=PAD):
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding is True:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])

                    if remaining_length > 0:
                        data[idx + seq_idx] = np.concatenate(
                            (data[idx + seq_idx], [padding_token] * remaining_length)
                        )

            batches.append(np.stack(data[idx : idx + batch_size]))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches


def pad(data):
    max_batch_length = 0

    # Get longest sentence in batch
    for seq in data:
        if len(seq) > max_batch_length:
            max_batch_length = len(seq)

    # Append X padding tokens until it reaches the max length
    for i, seq in enumerate(data):
        remaining_length = max_batch_length - len(seq)

        if remaining_length > 0:
            data[i] = np.concatenate((data[i], [PAD] * remaining_length))

    return data
