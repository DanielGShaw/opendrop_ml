
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle as pickle
import random
import numpy as np
import datetime
import time
import os
import glob
import shutil

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback # for early stopping
from tensorflow.keras.callbacks import ModelCheckpoint #for checkpoint saves
from tensorflow.keras.callbacks import TensorBoard #for checkpoint saves of history and figures
import warnings # for early stopping

import logging #for optuna logging
import sys #for optuna logging
import json #for history (backup logging)

import optuna #for hyperparameter optimisation
from optuna.integration import TFKerasPruningCallback
from optuna.trial import TrialState

# Define randomisation SEED globally for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)



def load_pkl_file(filename):
    with open(filename.numpy().decode("utf-8"), "rb") as f:
        data = pickle.load(f)

    inputs, labels = [], []
    for key, value in data.items():
        label = float(key.split('_')[0])  # Extract label from key
        labels.append(label)  # One label per sample
        inputs.append(value)  # One (50,2) input per sample

    inputs = np.array(inputs, dtype=np.float32)  # Shape (N, 50, 2)
    labels = np.array(labels, dtype=np.float32)  # Shape (N,)

    # Shuffle samples within each file using fixed seed
    indices = np.random.RandomState(SEED).permutation(len(labels))

    return inputs[indices], labels[indices]  # Keep full (N, 50,2) shape

def tf_load_pkl_file(filename):
    inputs, labels = tf.py_function(load_pkl_file, [filename], [tf.float32, tf.float32])

    # Ensure correct shape is set dynamically
    inputs.set_shape([None, 50, 2])  # Dynamic first dimension (batch size)
    labels.set_shape([None])         # Dynamic first dimension (batch size)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))

def load_dataset(batch_size):
    # Set variables
    cycle_length = 20
    block_length = 2

    # Get all .pkl file paths
    pkl_files = glob.glob("contour_dataset_5par/*.pkl")
    np.random.shuffle(pkl_files)  # Seeded shuffle to avoid ordering bias

    # Adjust train/validation/test split
    split_ratio_train = 0.65  # 65% train
    split_ratio_val = 0.15    # 15% validation
    split_ratio_test = 0.2    # 20% test

    train_idx = int(len(pkl_files) * split_ratio_train)
    val_idx = int(len(pkl_files) * (split_ratio_train + split_ratio_val))

    # Split dataset into train/validation/test
    train_files = pkl_files[:train_idx]
    val_files = pkl_files[train_idx:val_idx]
    test_files = pkl_files[val_idx:]

    # Create train dataset
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    train_ds = train_ds.interleave(
        lambda filename: tf_load_pkl_file(filename),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.shuffle(buffer_size=min(50000, len(train_files) * 10), seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    val_ds = val_ds.interleave(
        lambda filename: tf_load_pkl_file(filename),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.shuffle(buffer_size=min(20000, len(val_files) * 10), seed=SEED, reshuffle_each_iteration=False)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create test dataset
    test_ds = tf.data.Dataset.from_tensor_slices(test_files)
    test_ds = test_ds.interleave(
        lambda filename: tf_load_pkl_file(filename),
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # No shuffling for test set

    return train_ds, val_ds, test_ds

class EarlyStoppingWhenErrorLow(Callback):
    def __init__(self, monitor='val_mse', value=0.02, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        elif current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, history_path="training_history.json"):
        super(SaveHistoryCallback, self).__init__()
        self.history_path = history_path
        self.history_data = self._load_history()

    def _load_history(self):
        """Load existing training history from file."""
        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as f:
                return json.load(f)
        return {}

    def on_epoch_end(self, epoch, logs=None):
        """Save history at the end of each epoch."""
        if logs is None:
            logs = {}

        for key, value in logs.items():
            if key not in self.history_data:
                self.history_data[key] = []
            self.history_data[key].append(value)

        # Save to file
        with open(self.history_path, "w") as f:
            json.dump(self.history_data, f)

        print(f"Saved history after epoch {epoch + 1}")

def CA_activation(x):
    return tf.keras.activations.relu(x,max_value=180)

def objective(trial):
    # prep for info collection
    model_start_time = time.time()
    write = []
    write.append('TRAINING DATA\n')

    # trial hyperparameter suggestions
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 2e-3, log=True) #0.00011638101163629009
    batch_size = trial.suggest_int("batch_size", 16, 128, log=True) #42
    model_width = trial.suggest_categorical("model_width", [8,16, 32, 64, 128]) #32
    kernel_size = trial.suggest_int("kernel_size", 1,16)
    dense_size = trial.suggest_categorical("dense_size", [0,2,4,8,16,32,64,128])
    es_patience = trial.suggest_int("es_patience", 20, 2048, log=True) #512
    #es_patience = trial.suggest_int("es_patience", 1, 1, log=True)
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Metrics to be monitored by Optuna.
    monitor = "val_mean_squared_error"

    # Register the custom activation function in the custom_objects dictionary
    #custom_objects = {"CA_activation": CA_activation}

    # Create tf.keras model instance.
    #model, es_patience = create_model(trial) # no model to load, create new model
    #model, es_patience = load_model(trial) # load model from saved model file

    checkpointpath = './weights.best.tf'
    if os.path.exists(checkpointpath) == False:
        # Compose neural network with one hidden layer.
        model = Sequential([
            layers.Conv1D(model_width, kernel_size, padding='same', activation='relu'),
            layers.Conv1D(model_width*2, kernel_size, padding='same', activation='relu'),
            layers.Conv1D(model_width*4, kernel_size, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(128),#, activation=CA_activation),
            layers.Dense(1)
        ])
    else: #to continue training
        print('learning_rate: ', learning_rate)
        print('batch_size: ', batch_size)
        print('model_width: ', model_width)
        print('es_patience: ', es_patience)
        model = tf.keras.models.load_model(checkpointpath)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate),
        loss="mean_squared_error",
        #metrics=(['mean_absolute_error'],['mean_squared_error'])
        metrics=([tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
    )

    # Create dataset instance.
    train_ds, val_ds, test_ds = load_dataset(batch_size)
    print('dataset loaded...')

    #baseline = 0.02
    max_epochs = 100*es_patience
    #checkpointpath = "weights.best.tf"
    checkpoint = ModelCheckpoint(checkpointpath, monitor=monitor, verbose=0, save_best_only=True, mode='min')
    es = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=es_patience, verbose=0, mode="min", restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir=checkpointpath+'/logs')
    history_callback = SaveHistoryCallback(history_path=checkpointpath+"/training_history.json")

    #floor= EarlyStoppingWhenErrorLow(monitor=monitor, value=baseline, verbose=0)

    # Determine the starting epoch
    def get_last_epoch(history_path):
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history_data = json.load(f)
            
            # Get number of completed epochs from the length of any key
            if history_data:
                return len(next(iter(history_data.values()), []))  # Number of completed epochs
        return 0  # Start fresh if no history exists
    initial_epoch = get_last_epoch(checkpointpath+"/training_history.json")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=max_epochs,
        callbacks=[checkpoint,es,history_callback,tensorboard_callback]#floor,es]#,TFKerasPruningCallback(trial, monitor)]
    )    #floor= EarlyStoppingWhenErrorLow(monitor=monitor, value=baseline, verbose=0)

    # record info
    training_time = time.time() - model_start_time
    write.append("--- %s seconds ---" % training_time)
    mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    mse = history.history['mean_squared_error']
    val_mse = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # load best model
    model = tf.keras.models.load_model(checkpointpath)
    
    # copy best model to a save file
    score = min(history.history[monitor])
    score_dir = '_mse_'+str(score)
    shutil.copytree(checkpointpath, score_dir)
    
    with open(str(score_dir)+'/trainHistoryDict.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    # to load: history = pickle.load(open('/trainHistoryDict'), "rb")

    with open(str(score_dir)+'/modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    #model.summary(print_fn=myprint) # save the model output to txt file

    epochs_range = range(len(val_mae))

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(1, 3, 1,xscale='log',yscale='log',ylabel='Mean Absolute Error',xlabel='Epoch')
    ax1.plot(epochs_range, val_mae, label='Validation MAE')
    ax1.plot(epochs_range, mae, label='Training MAE')
    ax1.legend(loc='upper right')
    ax1.title.set_text('Training and Validation MAE')


    ax2 = plt.subplot(1, 3, 2,xscale='log',yscale='log',ylabel='Mean Squared Error',xlabel='Epoch')
    ax2.plot(epochs_range, val_mse, label='Validation MSE')
    ax2.plot(epochs_range, mse, label='Training MSE')
    ax2.legend(loc='upper right')
    ax2.title.set_text('Training and Validation MSE')

    ax3 = plt.subplot(1, 3, 3,xscale='log',yscale='log',ylabel='Loss',xlabel='Epoch')

    ax3.plot(epochs_range, val_loss, label='Validation Loss')
    ax3.plot(epochs_range, loss, label='Training Loss')
    ax3.legend(loc='upper right')
    ax3.title.set_text('Training and Validation Loss')

    fig.savefig(str(score_dir)+'/training_history.png') # save

    plt.tight_layout()

    plt.close()

    write.append('\nTEST DATA\n')
    evaluate = model.evaluate(test_ds,return_dict=True)
    write.append(evaluate)

    test_predictions = model.predict(test_ds).flatten()

    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1,2,1,ylabel='Predictions [angle, degrees]',xlabel='True Values [angle, degrees]',aspect='equal')
    ax1.title.set_text('Test set')
    ax1.scatter(test_keys, test_predictions, c='crimson')
    lims = [0,180]
    _ = plt.plot(lims, lims)

    ax2 = plt.subplot(1,2,2,ylabel='Predictions [angle, degrees]',xlabel='True Values [angle, degrees]',aspect='equal',
                     xlim = [min(test_keys)-1, max(test_keys)+1],ylim = [min(test_predictions)-1, max(test_predictions)+1])
    ax2.title.set_text('Test set')
    ax2.scatter(test_keys, test_predictions, c='crimson')
    lims = [min(test_keys), max(test_keys)]
    _ = plt.plot(lims, lims)

    plt.tight_layout()

    plt.savefig(str(score_dir)+'/test_data.png') # save

    plt.close()

    # output spread of test set error
    err = test_predictions-test_keys
    mu = np.mean(err)  # mean of distribution
    sigma = np.std(err)  # standard deviation of distribution
    num_bins = 50
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(err, num_bins, density=True) # the histogram of the data

    lower = np.mean(err) - (3*np.std(err))
    upper = np.mean(err) + (3*np.std(err))
    write.append('mean error is '+str.format('{0:.2e}',np.mean(err))+' and standard deviation is '+str.format('{0:.2e}',sigma))
    write.append('99.7% of errors are between '+str(lower)+' and '+(str(upper)))
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y, '--')
    # add mean and std deviation lines
    ax.axvline(mu,ymax=0.9,color='r')
    for n in [-1,1]:
        ax.axvline(mu+(n*sigma),ymax=0.68*0.9,color='r')
    for n in [-1,1]:
        ax.axvline(mu+(n*2*sigma),ymax=0.05*0.9,color='r')
    for n in [-1,1]:
        ax.axvline(mu+(n*3*sigma),ymax=0.01*0.9,color='r')
    ax.set_xlabel('Error')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of test set error: $\mu$='+str.format('{0:.2e}', mu)+', $\sigma$='+str.format('{0:.2e}', sigma))

    fig.tight_layout() # Tweak spacing to prevent clipping of ylabel
    plt.savefig(str(score_dir)+'/test_set_spread.png') #save

    plt.close()

    with open(str(score_dir)+'/outputted_data.txt','w') as f:
        for line in write:
            f.write(str(line))
            f.write('\n')

    print('Done')

    return score

def show_result(study):

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def main():

    print("Start time: "+str(datetime.datetime.now()))
    print('GPU information: ',tf.config.list_physical_devices('GPU'))

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    study_name='hyperparameter_study'
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        load_if_exists=True
    )

    # check if there is a failed trial
    #print('check for failed trials')
    #for trial in study.trials:
    #    if trial.state == optuna.trial.TrialState.FAIL:
    #        print('failed trial found')
    #        print(trial)
    #        study.enqueue_trial(trial.params, user_attrs={"memo": "loaded"})
    #print(study.trials)
    #study.enqueue_trial(study.trials[-1].params)
    
    params = {'learning_rate': 0.0001, 'batch_size': 42, 'model_width': 64, 'kernel_size': 10, 'dense_size': 32, 'es_patience': 20}#1789}
    study.enqueue_trial(params)

    study.optimize(objective, n_trials=1, timeout=(60*60*24)*2)#+(60*60*20)) #seconds * minutes * hours
    #2700 sec is 45 min, 21600 is 6 hours, 345600 is 4 days

    show_result(study)

if __name__ == "__main__":
    main()
