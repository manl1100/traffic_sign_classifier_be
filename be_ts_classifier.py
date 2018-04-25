import matplotlib.pyplot as plt
import random
import tensorflow as tf

import be_ts_dataset as ds

images28, labels = ds.load_data(ds.TRAIN_DATA_PATH)
test_images28, test_labels = ds.load_data(ds.TEST_DATA_PATH)

# Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Flatten input
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(
    inputs=images_flat,
    num_outputs=62,
    activation_fn=tf.nn.relu,
)

# Loss function
loss = tf.reduce_mean(
    input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits,
    )
)

# Optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(201):
        _, loss_val = sess.run(
            fetches=[train_op, accuracy],
            feed_dict={x: images28, y: labels},
        )

        if i % 10 == 0:
            print('Loss:', loss_val)

    # Testing model with random sample
    sample_indices = random.sample(range(len(images28)), 10)
    sample_images = [images28[i] for i in sample_indices]
    sample_labels = [labels[i] for i in sample_indices]

    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

    print('Labels:', sample_labels)
    print('Predicted:', predicted)

    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1+i)
        plt.axis('off')
        color = 'green' if prediction == truth else 'red'
        plt.text(
            x=40, y=10,
            s="Truth: {0}\nPrediction: {1}".format(truth, prediction),
            fontsize=12, color=color
        )

        plt.imshow(sample_images[i],  cmap="gray")

    plt.show()

    # Testing accuracy using test dataset
    predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
    match_count = sum(int(y == _y) for y, _y in zip(test_labels, predicted))
    accuracy = match_count / len(test_labels)

    print('Accuracy: {:.3f}'.format(accuracy))
