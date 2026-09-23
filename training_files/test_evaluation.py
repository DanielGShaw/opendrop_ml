
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

def do_this():
    # set up output file stuff
    write = []
    write.append('TEST DATA')
    score_dir = './_mse_0.4218772351741791/' 
    # load the trained model
    model = tf.keras.models.load_model(score_dir)

    # Create dataset instance.
    batch_size = 42
    train_ds, val_ds, test_ds = load_dataset(batch_size)
    print('dataset loaded...')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Separate inputs and labels in test_ds
    #test_inputs = []
    test_labels = []
    for batch in test_ds:  # Each batch is (input, label)
        inputs, labels = batch
        #test_inputs.append(inputs)
        test_labels.append(labels.numpy())

    # Convert lists of tensors to single tensors
    #test_inputs = tf.concat(test_inputs, axis=0)
    test_labels = tf.concat(test_labels, axis=0)

    #print('test_inputs.shape: ', test_input.shape)
    print('test_labels.shape: ', test_labels.shape)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # Evaluate
    evaluate = model.evaluate(test_ds,return_dict=True)
    write.append(evaluate)
    print('evaluation complete')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    #flatten() is causing memory issues, so perform in chunks
    test_predictions = model.predict(test_ds)
    #test_predictions = []
    #for batch in test_ds:
    #    batch_predictions = model.predict(batch[0])  # Predict only on inputs
    #    test_predictions.append(batch_predictions)
    print('predictions complete')
    print('test_predictions[0].shape: ',test_predictions[0].shape)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    test_predictions = np.concatenate(test_predictions).squeeze()  # Convert to a single array
    print('predictions concatenated')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    if 1: #checks
        print('test_labels.shape: ',test_labels.shape)
        print('test_predictions.shape: ', test_predictions.shape)
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1,2,1,ylabel='Predictions [angle, degrees]',xlabel='True Values [angle, degrees]',aspect='equal')
    ax1.title.set_text('Test set')
    ax1.scatter(test_labels, test_predictions, c='crimson')
    lims = [0,180]
    _ = plt.plot(lims, lims)

    ax2 = plt.subplot(1,2,2,ylabel='Predictions [angle, degrees]',xlabel='True Values [angle, degrees]',aspect='equal',
                     xlim = [min(test_labels)-1, max(test_labels)+1],ylim = [min(test_predictions)-1, max(test_predictions)+1])
    ax2.title.set_text('Test set')
    ax2.scatter(test_labels, test_predictions, c='crimson')
    lims = [min(test_labels), max(test_labels)]
    _ = plt.plot(lims, lims)

    plt.tight_layout()

    plt.savefig(str(score_dir)+'/test_data.png') # save

    plt.close()

    print('test_data.png image completed')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # output spread of test set error
    err = test_predictions-test_labels
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
    
    print('test_set_spread.png image completed')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

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

    do_this()

if __name__ == "__main__":
    main()
