import os
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import numpy as np
import tqdm

(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x = np.reshape(x, (-1, 784))/255
x_test = np.reshape(x_test, (-1, 784))/255

y = tf.one_hot(y, 10).numpy()
y_test = tf.one_hot(y_test, 10).numpy()

(x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x = np.reshape(x, (-1, 3072))/255
x_test = np.reshape(x_test, (-1, 3072))/255

y = tf.one_hot(tf.reshape(y, (-1,)), 10).numpy()
y_test = tf.one_hot(tf.reshape(y_test, (-1,)), 10).numpy()


class MoE(tf.keras.layers.Layer):

    def __init__(self, n_in, n_out, n_experts, k=1, expert_model=None, selector_model=None):
        super().__init__()

        self.n_in, self.n_out, self.n_experts = n_in, n_out, n_experts
        self.k = k

        if expert_model is None:
            def expert_model():
                return tf.keras.models.Sequential([tf.keras.layers.Input(self.n_in), tf.keras.layers.Dense(n_out, 'relu')])

        if selector_model is None:
            def selector_model():
                return tf.keras.models.Sequential([tf.keras.layers.Input(self.n_in), tf.keras.layers.Dense(n_experts, 'softmax')])

        self.experts = [expert_model() for _ in range(self.n_experts)]
        self.selector = selector_model()


    def cv_squared(self, x, eps=1e-10):
        return tf.math.reduce_variance(x) / (tf.reduce_mean(x)**2 + eps)


    def call(self, inputs):
        shape = inputs.shape
        gate_odds = self.selector(inputs)

        aux_loss = self.cv_squared(tf.reduce_sum(gate_odds, axis=-1))

        values, indices, mask = gate(gate_odds, self.k)
        origins_expert = tf.stack([tf.range(self.k) for _ in range(shape[0])], axis=0)

        sample_to_expert = [[] for _ in range(self.n_experts)]
        sample_to_opinion = [[] for _ in range(self.n_experts)]

        for sample, selection in enumerate(indices):
            for opinion_x, to_expert in enumerate(selection): # opinion_x is a play on second opinion as there are a fuck tonne of expert, don't hate me, naming variabels is hard. And no that wasn't a spelling error, variables is a creative variable name.
                sample_to_expert[to_expert] += [sample]
                sample_to_opinion[to_expert] += [int(origins_expert[sample, opinion_x])]


        y = [None for _ in range(self.n_experts)]
        for i in range(self.n_experts):
            if len(sample_to_expert[i]) != 0:
                y[i] = self.experts[i](tf.gather(inputs, sample_to_expert[i]))
        # y = [self.experts[i](tf.gather(inputs, sample_to_expert[i])) for i in range(self.n_experts)]
        output_placeholder = tf.zeros(shape=(shape[0],) + (self.k,) + shape[1:-1] + (self.n_out,))


        for i, y_expert in enumerate(y):
            if y_expert is not None:
                indices_batch = tf.stack([tf.constant(sample_to_expert[i]), tf.constant(sample_to_opinion[i])], axis=1)
                output_placeholder = tf.tensor_scatter_nd_add(output_placeholder, indices_batch, y_expert)
        y = output_placeholder * tf.reshape(values, values.shape + (1,))
        y = tf.reduce_sum(y, axis=-2)
        return y, aux_loss


class MoE_net(tf.keras.layers.Layer):

    def __init__(self, n_in, n_out, n_experts, n_layers, k=1, expert_model=None, selector_model=None):
        super().__init__()

        self.layer = MoE(n_in, n_out, n_experts, k, expert_model=expert_model, selector_model=selector_model)
        self.n_layers = n_layers




    def call(self, inputs):

        aux_loss = 0
        for _ in range(self.n_layers):
            inputs, extra_loss = self.layer(inputs)
            aux_loss += extra_loss

        y = inputs

        return y, aux_loss



def gate(gate_odds, k):
    values, indices = tf.math.top_k(gate_odds, k)
    mask = tf.zeros_like(gate_odds)
    depth = gate_odds.shape[-1]
    for i in range(k):
        mask += tf.one_hot(indices[..., i], depth=depth)
    return values, indices, mask











# in_layer = tf.keras.layers.Dense(50, "relu")
# layer_1 = MoE(20, 20, 100, k=3)
# layer_2 = MoE(20, 20, 100, k=3)
model = MoE_net(3072, 100, 20, 1, k=3)
out = tf.keras.layers.Dense(10, "softmax")


optimizer, loss = tf.keras.optimizers.Adam(1e-3), tf.keras.losses.categorical_crossentropy
acc = tf.keras.metrics.categorical_accuracy

# model = tf.keras.models.Sequential([in_layer, layer_1, layer_2, out])

for epoch in range(100):
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    batch_size = 64

    batches = len(x) // batch_size

    p_bar = tqdm.tqdm(np.array_split(x, batches))
    for x_batch, y_batch in zip(p_bar, np.array_split(y, batches)):
        with tf.GradientTape() as tape:
            aux_loss = 0
            # a = in_layer(x_batch)
            a, l1 = model(x_batch)
            aux_loss += l1*.01
            a = out(a)
            cat_loss = tf.reduce_sum(loss(y_batch, a))
            l = cat_loss + aux_loss
            p_bar.set_description(f"acc: {round(float(tf.reduce_sum(acc(y_batch, a)) / len(x_batch)), 2)}, loss: {round(float(cat_loss / len(x_batch)), 2)}", refresh=True)

        vars = \
            model.trainable_variables + out.trainable_variables
        grads = tape.gradient(l, vars)
        optimizer.apply_gradients(zip(grads, vars))

    a, _ = model(x_test)
    a = out(a)
    print(tf.reduce_mean(acc(y_test, a)))


