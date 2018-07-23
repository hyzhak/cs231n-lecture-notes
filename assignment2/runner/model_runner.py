import tensorflow as tf
import numpy as np
import math
import time
import matplotlib.pyplot as plt


def run_model(sess, X, y, is_training, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False, learning_rate=None, learning_rate_value=10e-3, part_of_dataset=1.0,
              snapshot_name=None,
              ):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    all_losses = []
    all_correct = []
    # counter
    iter_cnt = 0

    saver = tf.train.Saver()
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size * part_of_dataset))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {
                X: Xd[idx, :],
                y: yd[idx],
                is_training: training_now,
            }
            if learning_rate is not None:
                if isinstance(learning_rate_value, float):
                    feed_dict[learning_rate] = learning_rate_value
                elif isinstance(learning_rate_value, list):
                    feed_dict[learning_rate] = learning_rate_value[e]
                else:
                    raise Error('unsupported learning_rate, valid are list or float')
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = sess.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            all_correct.append(np.sum(corr) / actual_batch_size)
            all_losses.append(loss)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()

        if training_now and snapshot_name is not None:
            save_path = saver.save(sess, f'./snapshots/{snapshot_name}/model')
            print(f'Model saved in path: {save_path}')
    return total_loss, total_correct, all_losses, all_correct
