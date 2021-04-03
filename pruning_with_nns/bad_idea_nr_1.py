import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tqdm
import numpy as np
import math
import time


def decay(rewards, decay_factor):
    """
    Berekent de echte rewards aan de hand van de verkregen rewards van een episode op elk tijdstip en een decay_factor

    :param rewards: een array/list met rewards per stap
    :param decay_factor: getal tussen 0 en 1 dat het belang van de toekomst aangeeft
    :return: een array met rewards waar de toekomst WEL in mee is genomen

    VB: decay([1, 0, 1], .9) --> [1.81, .9, 1]
    """
    decayed_rewards = np.zeros(len(rewards))
    decayed_rewards[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        decayed_rewards[i] = rewards[i] + decay_factor * decayed_rewards[i + 1]
    return decayed_rewards


def decay_and_normalize(total_rewards, decay_factor):
    """
    Past decay toe op een batch van episodes en normaliseert over het geheel

    :param total_rewards: list van lists/arrays, waar de inner lists rewards bevatten
    :param decay_factor: getal tussen 0 en 1 dat het belang van de toekomst aangeeft
    :return: één nieuwe array met nieuwe rewards waar de toekomst in mee is genomen en die genormaliseerd is

    VB: decay_and_normalize([[0, 1], [1, 1, 1]], .9)
        eerst decay --> [[.9, 1], [2.71, 1.9, 1]]
        dan normaliseren --> [-0.85, -0.71, 1.71, 0.56, -0.71]
    """
    for i, rewards in enumerate(total_rewards):
        total_rewards[i] = decay(rewards, decay_factor)
    total_rewards = np.concatenate(total_rewards)
    return (total_rewards - np.mean(total_rewards)) / np.std(total_rewards)


def categorical_crossentropy(y_true, y_pred):
    y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1 - 1e-7)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


# model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, 3, 2, activation="relu", padding="same"),
#                                     tf.keras.layers.Conv2D(64, 3, 2, activation="relu", padding="same"),
#                                     tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(200, "relu"),
#                                     tf.keras.layers.Dense(10, "softmax")])
prune_net = tf.keras.models.Sequential([tf.keras.layers.Dense(10, "relu"),
                                        tf.keras.layers.Dense(10, "relu"),
                                        tf.keras.layers.Dense(1, "sigmoid")])

prune_optimizer = tf.keras.optimizers.Adam(3e-4)

(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# x = x.reshape((-1, 28, 28, 1))/255
# x_test = x_test.reshape((-1, 28, 28, 1))/255
x = x.reshape((-1, 784)) / 255
x_test = x_test.reshape((-1, 784)) / 255

y = tf.one_hot(y, 10).numpy()
y_test = tf.one_hot(y_test, 10).numpy()


def attempt(x, y, x_test, y_test, epochs, prune_p=.05, randomness=.0):
    layers = [tf.keras.layers.Dense(200, "relu"), tf.keras.layers.Dense(80, "relu"),
              tf.keras.layers.Dense(10, "softmax")]

    model = tf.keras.models.Sequential([tf.keras.layers.Input(784)] + layers)
    model.build()

    params = sum([np.product(layer.weights[0].shape) for layer in layers])

    optimizer = tf.keras.optimizers.Adam(3e-4)

    p_left = 1

    batch_size = 256

    acc_list = []
    inputs_prune = []
    outputs_prune = []
    blocked_prune = []

    activated = [np.ones((layer.weights[0].shape)) == 1 for layer in layers]
    for epoch in range(epochs):

        blocked_prune += [np.concatenate([activate.flatten() for activate in activated], axis=0).flatten()]

        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        batches = math.ceil(len(x) / batch_size)
        x_batches = np.array_split(x, batches)
        y_batches = np.array_split(y, batches)

        loss_arr = np.zeros(batches)
        acc_arr = np.zeros(batches)
        count_arr = np.zeros(batches)

        # p_bar = tqdm.trange(batches)
        for batch in range(batches):
            x_in = x_batches[batch]
            y_out = y_batches[batch]

            intermediate = [x_in]

            with tf.GradientTape() as tape:
                for layer in layers:
                    x_in = layer(x_in)
                    intermediate += [x_in]

                loss = categorical_crossentropy(y_out, x_in)
                loss = tf.reduce_sum(loss)

            train_vars = model.trainable_variables
            grads = tape.gradient(loss, train_vars)
            optimizer.apply_gradients(zip(grads, train_vars))

            loss_arr[batch] = loss
            acc_arr[batch] = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(y_out, x_in))
            count_arr[batch] = len(x_in)

            # p_bar.set_description(f"Loss: {np.sum(loss_arr[loss_arr != 0]) / np.sum(count_arr[count_arr != 0])}, "
            #                       f"acc: {np.sum(acc_arr[acc_arr != 0]) / np.sum(count_arr[count_arr != 0])}")

        new_prune_p = prune_p * p_left * (1 - randomness)
        p_left -= new_prune_p

        batch = intermediate[0].shape[0]

        odds = []

        input_shard = []

        for i, layer in enumerate(layers):
            weights = layer.weights[0]
            shape = weights.shape

            a = np.tile(np.array(intermediate[i]).reshape((-1, shape[0], 1)), (1, 1, shape[1])).reshape((-1, 1))
            b = np.tile(np.array(intermediate[i + 1]).reshape((-1, 1, shape[1])), (1, shape[0], 1)).reshape((-1, 1))
            c = np.tile(weights.numpy().reshape((-1, shape[0], shape[1])), (batch, 1, 1)).reshape((-1, 1))
            inputs = np.concatenate([a, b, c], axis=1)
            input_shard += [inputs]

            pred = tf.reduce_mean(tf.reshape(prune_net(inputs), (batch,) + layer.weights[0].shape), axis=0).numpy()
            pred[~activated[i]] = 1

            pred = pred.flatten()
            odds += [pred]

        inputs_prune += [np.concatenate(input_shard, axis=0)]

        output_shard = []

        shaving_n = int(new_prune_p * params * (1 - randomness))

        flat_odds = np.concatenate(odds, axis=0)

        flat_odds = np.sort(flat_odds.flatten())
        odd = flat_odds[shaving_n]
        for i, layer in enumerate(layers):
            l_params = np.product(layer.weights[0].shape)
            pruned = (odds[i] > odd).reshape((-1,))
            possible_randos = pruned & activated[i].reshape((-1,))
            if np.sum(possible_randos) != 0:
                razors = int(l_params * randomness * prune_p * np.sum(possible_randos) / np.product(possible_randos.shape))
                randos = np.random.choice(np.arange(len(possible_randos)), razors, False,
                                          p=possible_randos / np.sum(possible_randos))
                pruned[randos] = False
            output_shard += [pruned]
            activated[i] = activated[i] & pruned.reshape(activated[i].shape)
            weights = layer.weights[0].numpy()
            weights = weights * activated[i]
            layer.weights[0].assign(weights)

            # print(f"shaved {np.count_nonzero(activated[i]) / l_params}")

        outputs_prune += [np.concatenate(output_shard, axis=0)]

        # time.sleep(1000)

        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        y_test = y_test[indices]
        batches = math.ceil(len(x_test) / batch_size)
        x_batches = np.array_split(x_test, batches)
        y_batches = np.array_split(y_test, batches)

        loss_arr = np.zeros(batches)
        acc_arr = np.zeros(batches)
        count_arr = np.zeros(batches)

        # p_bar = tqdm.trange(batches)
        for batch in range(batches):
            x_in = x_batches[batch]
            y_out = y_batches[batch]

            output = model(x_in)
            loss = categorical_crossentropy(y_out, output)
            loss = tf.reduce_sum(loss)

            loss_arr[batch] = loss
            acc_arr[batch] = tf.reduce_sum(tf.keras.metrics.categorical_accuracy(y_out, output))
            count_arr[batch] = len(x_in)

            # p_bar.set_description(f"Val_loss: {np.sum(loss_arr[loss_arr != 0]) / np.sum(count_arr[count_arr != 0])}, "
            #                       f"val_acc: {np.sum(acc_arr[acc_arr != 0]) / np.sum(count_arr[count_arr != 0])}")
        acc_list += [np.sum(acc_arr[acc_arr != 0]) / np.sum(count_arr[count_arr != 0])]
    return inputs_prune, outputs_prune, blocked_prune, acc_list


epochs = 10
runs = 10
batches = 10000

steps = 10000

for step in range(steps):
    runs_ran = 0
    expected = np.zeros(epochs)
    inputs, outputs, blocked, reward = [], [], [], []
    for _ in range(runs):
        i, o, b, r = attempt(x, y, x_test, y_test, epochs, randomness=.1, prune_p=.05)
        inputs += [i]
        outputs += [o]
        blocked += [b]
        reward += [r]
        print(step, _, r)
    reward = decay_and_normalize(reward, .9)
    expected = runs_ran * expected + np.sum(reward.reshape((-1, epochs)), axis=0)
    runs_ran += runs
    expected /= runs_ran
    reward = reward.reshape((-1, epochs)) - expected
    reward = reward / np.std(reward)

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    blocked = np.array(blocked)
    reward = np.array(reward).reshape((runs, epochs, 1))

    batch_size = int(inputs.shape[2] / outputs.shape[2])

    outputs = np.tile(outputs, (1, 1, batch_size))
    blocked = np.tile(blocked, (1, 1, batch_size))
    reward = np.tile(reward, (1, 1, inputs.shape[2]))

    inputs = inputs.reshape((-1, 3))
    outputs = outputs.reshape((-1, 1))
    blocked = blocked.reshape((-1, 1))
    reward = reward.reshape((-1, 1))

    indices = np.arange(len(inputs))
    np.random.shuffle(indices)

    inputs = inputs[indices]
    outputs = outputs[indices]
    blocked = blocked[indices]
    reward = blocked[indices]

    inputs = np.array_split(inputs, batches)
    outputs = np.array_split(outputs, batches)
    blocked = np.array_split(blocked, batches)
    reward = np.array_split(reward, batches)

    for net_in, net_out, block, adv in zip(inputs, outputs, blocked, reward):
        with tf.GradientTape() as tape:
            output = prune_net(net_in)
            loss = tf.reduce_sum(tf.keras.losses.mse(net_out, output)) / len(net_in) ** .5

        train_vars = prune_net.trainable_variables
        grads = tape.gradient(loss, train_vars)
        prune_optimizer.apply_gradients(zip(grads, train_vars))

    tf.keras.models.save_model(prune_optimizer, "C:/tensorTestModels/prune_learner")

    print(inputs.shape, outputs.shape, blocked.shape, reward.shape)