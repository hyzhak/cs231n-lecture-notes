from runner.model_runner import run_model
from runner.num_of_trainable import num_of_trainable
import tensorflow as tf
import time


def estimate_model(model, X_train, y_train, X_val, y_val):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
    y = tf.placeholder(tf.int64, [None], name='y')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    is_training = tf.placeholder(tf.bool)

    y_out, params_to_eval = model['model_builder'](X, y, is_training, **model)

    mean_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
        onehot_labels=tf.one_hot(y, y_out.shape[1]),
        logits=y_out,
    ))

    #     optimizer = tf.train.RMSPropOptimizer(
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
    )

    # batch normalization in tensorflow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)

    num_of_trainable_val = num_of_trainable()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('Training')
        train_time = time.perf_counter()
        total_loss, total_correct, all_losses, all_correct = run_model(
            sess,
            X, y, is_training, y_out, mean_loss,
            X_train, y_train,
            model.get('num_of_epochs', 2), batch_size=64, print_every=100, training=train_step, plot_losses=True,
            learning_rate=learning_rate,
            learning_rate_value=model.get('learning_rate', 1e-3),
            part_of_dataset=model.get('part_of_dataset', 1.0),
            snapshot_name=f'{model["group"]}/{model["name"]}'.replace(' ', '-').lower(),
        )
        train_time = time.perf_counter() - train_time
        print('training time (seconds)', train_time)
        training = {
            'total_lost': total_loss,
            'total_correct': total_correct,
            'losses': all_losses,
            'accuracy': all_correct,
            'time': train_time,
        }

        print('Validation')
        validation_time = time.perf_counter()
        total_loss, total_correct, all_losses, all_total_correct = run_model(
            sess,
            X, y, is_training, y_out, mean_loss,
            X_val, y_val,
            epochs=1, batch_size=64,
            learning_rate=learning_rate,
            learning_rate_value=model.get('learning_rate', 1e-3),
        )
        validation_time = time.perf_counter() - validation_time
        print('validation time (seconds)', validation_time)

        validation = {
            'total_lost': total_loss,
            'total_correct': total_correct,
            'losses': all_losses,
            'accuracy': all_correct,
            'time': validation_time,
        }

        # estimate how much time it would get to predict
        print('predict')
        validation_time = time.perf_counter()
        run_model(
            sess,
            X, y, is_training, y_out, mean_loss,
            X_val[:1], y_val[:1],
            epochs=1, batch_size=64,
            learning_rate=learning_rate,
            learning_rate_value=model.get('learning_rate', 1e-3),
        )
        validation_time = time.perf_counter() - validation_time

        predict = {
            'time': validation_time,
        }

        # eval params and store
        params = {}
        for p in params_to_eval:
            params[p] = sess.run(p, feed_dict={
                X: X_val[:1],
                y: y_val[:1],
                is_training: False,
            })


    return {
        'params': params,
        'hyper_params': model,
        'res': {
            'predict': predict,
            'training': training,
            'validation': validation,
            'num_of_trainable': num_of_trainable_val,
        },
    }
