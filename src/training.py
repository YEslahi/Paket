import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import MobileNet as MOBILENET
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import utility
import datetime
from sklearn.metrics import classification_report

def train(input_params, train, test, valid, class_cnt):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # tensorboard
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    valid_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # todo: create model with hyperparams with model_dir = '../data/models/params/current_time/'
    model_dir = '../data/models/model-' + current_time
    # Instantiate an optimizer.
    optimizer = Adam(learning_rate=0.001)
    # Instantiate a loss function.
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    train_step = test_step = 0

    # Prepare the metrics.
    #todo use same variable for all the acc_metrics.
    acc_metric = SparseCategoricalAccuracy()

    if utility.dir_empty(model_dir):
        # model definition
        mobilenet = MOBILENET(include_top=False,
                              input_shape=(224, 224, 3),
                              weights='imagenet',
                              pooling='avg',
                              dropout=0.001)
        mobilenet.summary()
        # select till which layer use mobilenet.
        base_model = Model(inputs=mobilenet.input, outputs=mobilenet.output)
        base_model.summary()

        model = Sequential([
            base_model,
            Dropout(0.2),
            Dense(units=class_cnt, activation='softmax'),
        ])
        model.summary()

        epochs = 200
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            for batch_idx, (x_batch_train, y_batch_train) in enumerate(train):
                with tf.GradientTape() as tape:
                    # forward pass
                    logits = model(x_batch_train, training=True)

                    # compute loss for mini batch
                    loss_value = loss_fn(y_batch_train, logits)

                grads = tape.gradient(loss_value, model.trainable_weights)

                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                acc_metric.update_state(y_batch_train, logits)

                with train_summary_writer.as_default():
                    # import code; code.interact(local=dict(globals(), **locals()))
                    #TODO: add the metrics for test too.
                    #TODO: take the mean of the losses in every batch and then show,
                    #TODO       loss_value is last loss of the batch(only 1).
                            
                    tf.summary.scalar('loss', loss_value, step=train_step)
                    tf.summary.scalar('accuracy', acc_metric.result(), step=train_step)
                    train_step += 1

                if batch_idx % 10 == 0:
                    print("training loss for one batch at step %d: %.4f" % (batch_idx, float(loss_value)))
            # Display metrics at the end of each epoch.
            
            print("Training acc over epoch: %.4f" % (float(acc_metric.result()),))

            # Reset training metrics at the end of each epoch
            acc_metric.reset_states()


            # iterate on validation 
            for batch_idx, (x_batch_val, y_batch_val) in enumerate(valid):
                # val_logits: y_pred of the validation. 
                val_logits = model(x_batch_val, training=False)
                loss = loss_fn(y_batch_val, val_logits)
                # Update val metrics
                acc_metric.update_state(y_batch_val, val_logits)

                with valid_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=test_step)
                    tf.summary.scalar('accuracy', acc_metric.result(), step=test_step)
                    test_step += 1
                
            print("Validation acc: %.4f" % (float(acc_metric.result()),))
            # print(classification_report(y_batch_val, val_logits, target_names=labels))
            acc_metric.reset_states()
        
        acc_metric.reset_states()
        model.save(model_dir + 'model')
        
    else:  # if model_dir is not empty
        print("model already exist. loading model...")
        model = load_model(model_dir+'model')

