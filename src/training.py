import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import MobileNet as MOBILENET
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import utility
import datetime

def train(input_params, train, test, valid, class_cnt):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # todo: create model with hyperparams with model_dir = '../data/models/params/current_time/'
    model_dir = '../data/models/1/'
    # Instantiate an optimizer.
    optimizer = Adam(learning_rate=0.001)
    # Instantiate a loss function.
    loss_fn=SparseCategoricalCrossentropy(from_logits=True)
    # Prepare the metrics.
    train_acc_metric = SparseCategoricalAccuracy()
    val_acc_metric = SparseCategoricalAccuracy()

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

        epochs = 2
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(train):
                with tf.GradientTape() as tape:
                    # forward pass
                    logits = model(x_batch_train, training=True)

                    # compute loss for mini batch
                    loss_value = loss_fn(y_batch_train, logits)

                grads = tape.gradient(loss_value, model.trainable_weights)

                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)

                if step % 10 == 0:
                    print("training loss for one batch at step %d: %.4f" % (step, float(loss_value)))
            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
        model.save(model_dir + 'model')
        
    else:  # if model_dir is not empty
        print("model already exist. loading model...")
        model = load_model(model_dir+'model')

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in valid:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    
