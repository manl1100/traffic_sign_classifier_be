import tensorflow as tf
import be_ts_dataset as ds

images28, labels = ds.load_training_data()

# Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
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

    for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run(
            fetches=[train_op, accuracy],
            feed_dict={x: images28, y: labels},
        )

        if i % 10 == 0:
            print('Loss:', loss)
        print('DONE WITH EPOCH')
