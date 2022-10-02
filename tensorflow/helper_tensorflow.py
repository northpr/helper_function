import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import zipfile

## Callbacks ##
# LearningRateScheduler
def callbacks_learning_rate(epoch):
  """
  Finding ideal learing rate by change learning rate.
  Need to plot the learning rate vs loss plot to find the best value
  Example: plot_lr_loss(epochs=10)
  -> history_8 = model_8.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels),
                        callbacks=[callbacks_learning_rate(epoch=10)])
  """
  callbacks_lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
  return callbacks_lr_scheduler


# Early Stopping
def callbacks_early_stopping(monitor: str, patience: int, verbose=0):
  """
  Early stop if the model is not improve.
  Example: 
  - For classification model: callbacks_early_stopping(monitor="accuracy", patience=20, verbose=1)
  - For regression model: callbacks_early_stopping(monitor="mae", patience=100, verbose=1)
  -> history_8 = model_8.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels),
                        callbacks=[callbacks_early_stopping(monitor="accuracy", patience=5)])
  """
  callbacks_early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=verbose, restore_best_weights=True)
  return callbacks_early_stopping

# Tensorboard callback
def callbacks_create_tensorboard(dir_name: str, experiment_name: str):
  """
  Example: callbacks=[callback_create_tensorboard(dir_name="tensorflow_hub",
                                                  experiment_name="efficientnet")])
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

def callbacks_model_checkpoint(checkpoint_path: str, monitor: str="val_accuracy", save_best_only=True, save_weights_only=True, verbose=0):
  """
  Example:
  - callbacks_model_checkpoint("model_checkpoints/cp.ckpt")
  Args:
      checkpoint_path (str): _description_
      monitor (str, optional): _description_. Defaults to "val_accuracy".
      save_best_only (bool, optional): _description_. Defaults to True.
      save_weights_only (bool, optional): _description_. Defaults to True.
      verbose (int, optional): _description_. Defaults to 0.
  """
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        monitor=monitor,
                                                        save_best_only=save_best_only,
                                                        save_weights_only=save_weights_only,
                                                        verbose=verbose)

  return model_checkpoint

# ================================ #


## Plot graph ##
# Plot loss vs learning rate curve
def plot_lr_loss(history, figsize=(10,7)):
  """
  * Must use with callbacks_learning_rate() *
  Plot the learning rate versus the loss by getting history from the model training.
  """
  lrs = 1e-4 * (10 ** (np.arange(100)/20))
  plt.figure(figsize=figsize)
  plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
  plt.xlabel("Learning Rate")
  plt.ylabel("Loss")
  plt.title("Learning rate vs. loss")

# Plot the validation and training data separately
def plot_loss_curves(history,figsize=(7,7)):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  plt.figure(figsize=figsize)
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
  
# ================================ #

## Evaluate function ##
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, precision_recall_fscore_support

def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (assuming no seasonality of data).
  """
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))

  # Find MAE of naive forecast (no seasonality)
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # our seasonality is 1 day (hence the shifting of 1 day)

  return mae / mae_naive_no_season

def evaluate_classification(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

def evaluate_regression(y_true, y_pred, mape=False, mase=False):

  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  
  # Calculate multiple metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = (tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred) \
    if mape == True else ["No value"])
  mase = (mean_absolute_scaled_error(y_true, y_pred) \
    if mase  == True else ["No value"])
  
  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}
# ================================ #


## Zip ##

def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()


def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")