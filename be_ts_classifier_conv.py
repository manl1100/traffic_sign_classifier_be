import matplotlib.pyplot as plt
import random
import tensorflow as tf

import be_ts_dataset as ds

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model(features, labels, mode):
    # Input layer
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
    )

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    _, n2, n3, n4 = pool2.get_shape().as_list()
    pool2_flat = tf.reshape(pool2, [-1, n2 * n3 * n4])

    dense = tf.layers.dense(
        inputs=pool2_flat, units=62, activation=tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(inputs=dropout, units=62)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=62)
    # loss = tf.losses.softmax_cross_entropy(
    #     onehot_labels=onehot_labels,
    #     logits=logits
    # )
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def main(unused_argv):
    # Train data
    images28, labels = ds.load_data(ds.TRAIN_DATA_PATH)

    # Test data
    test_images28, test_labels = ds.load_data(ds.TEST_DATA_PATH)

    # Estimator
    traffic_sign_classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir='/tmp/traffic_sign_convnet_model'
    )

    # Set up logging for predictions
    tensors_to_log = {
        'probabilities': 'softmax_tensor'
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # Train model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': images28},
        y=labels,
        # batch_size=-1,
        num_epochs=None,
        shuffle=True
    )

    traffic_sign_classifier.train(
        input_fn=train_input_fn,
        steps=50000,
        hooks=[logging_hook]
    )

    # Test model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_images28},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = traffic_sign_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    # >> {'accuracy': 0.85833335, 'loss': 0.5191961, 'global_step': 30400}
    # >> {'accuracy': 0.9253968, 'loss': 0.2948123, 'global_step': 80400}


if __name__ == '__main__':
    tf.app.run()
