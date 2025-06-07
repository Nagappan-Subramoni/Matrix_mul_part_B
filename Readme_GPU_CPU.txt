GPU available: 1 GPU(s) found
GPU names: ['/physical_device:GPU:0']
Using device: GPU
Loading datasets...
Found 2000 files belonging to 2 classes.
2025-06-07 17:39:57.111995: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-06-07 17:39:57.656630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 5563 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9
Found 1000 files belonging to 2 classes.
Found 2000 files belonging to 2 classes.
Creating base model...

==================================================
Training Base Model
==================================================
Epoch 1/10
2025-06-07 17:39:58.959384: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8907
2025-06-07 17:39:59.651008: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
63/63 [==============================] - 6s 66ms/step - loss: 0.7490 - accuracy: 0.5175 - val_loss: 0.6842 - val_accuracy: 0.5360
Epoch 2/10
63/63 [==============================] - 3s 51ms/step - loss: 0.7044 - accuracy: 0.5345 - val_loss: 0.6735 - val_accuracy: 0.5180
Epoch 3/10
63/63 [==============================] - 3s 51ms/step - loss: 0.7629 - accuracy: 0.5865 - val_loss: 0.6482 - val_accuracy: 0.6120
Epoch 4/10
63/63 [==============================] - 3s 50ms/step - loss: 0.6445 - accuracy: 0.6400 - val_loss: 0.9186 - val_accuracy: 0.5150
Epoch 5/10
63/63 [==============================] - 3s 51ms/step - loss: 0.6130 - accuracy: 0.6730 - val_loss: 0.6253 - val_accuracy: 0.6610
Epoch 6/10
63/63 [==============================] - 3s 52ms/step - loss: 0.5961 - accuracy: 0.6830 - val_loss: 0.6844 - val_accuracy: 0.6490
Epoch 7/10
63/63 [==============================] - 3s 50ms/step - loss: 0.5433 - accuracy: 0.7240 - val_loss: 0.6527 - val_accuracy: 0.6580
Epoch 8/10
63/63 [==============================] - 3s 51ms/step - loss: 0.5206 - accuracy: 0.7485 - val_loss: 0.6284 - val_accuracy: 0.6640
Epoch 9/10
63/63 [==============================] - 3s 51ms/step - loss: 0.4854 - accuracy: 0.7715 - val_loss: 0.6161 - val_accuracy: 0.7100
Epoch 10/10
63/63 [==============================] - 3s 51ms/step - loss: 0.4484 - accuracy: 0.7940 - val_loss: 0.7022 - val_accuracy: 0.7170
Training time: 35.71 seconds
63/63 [==============================] - 1s 16ms/step - loss: 0.7464 - accuracy: 0.7080
Test accuracy: 0.708

Creating augmented model with dropout...
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.

==================================================
Training Augmented Model
==================================================
Epoch 1/10
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting RngReadAndSkip cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting Bitcast cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting StatelessRandomUniformV2 cause there is no registered converter for this op.
WARNING:tensorflow:Using a while_loop for converting ImageProjectiveTransformV3 cause there is no registered converter for this op.
63/63 [==============================] - 15s 198ms/step - loss: 0.7299 - accuracy: 0.5330 - val_loss: 1.2167 - val_accuracy: 0.5000
Epoch 2/10
63/63 [==============================] - 12s 185ms/step - loss: 0.7085 - accuracy: 0.5470 - val_loss: 0.6768 - val_accuracy: 0.6270
Epoch 3/10
63/63 [==============================] - 12s 184ms/step - loss: 0.6930 - accuracy: 0.6135 - val_loss: 0.6583 - val_accuracy: 0.6090
Epoch 4/10
63/63 [==============================] - 12s 186ms/step - loss: 0.6371 - accuracy: 0.6495 - val_loss: 0.7001 - val_accuracy: 0.5890
Epoch 5/10
63/63 [==============================] - 11s 181ms/step - loss: 0.6256 - accuracy: 0.6575 - val_loss: 0.6360 - val_accuracy: 0.6270
Epoch 6/10
63/63 [==============================] - 11s 180ms/step - loss: 0.6012 - accuracy: 0.6705 - val_loss: 0.7549 - val_accuracy: 0.5660
Epoch 7/10
63/63 [==============================] - 12s 183ms/step - loss: 0.6017 - accuracy: 0.6805 - val_loss: 0.5617 - val_accuracy: 0.6820
Epoch 8/10
63/63 [==============================] - 11s 179ms/step - loss: 0.5888 - accuracy: 0.7045 - val_loss: 0.5647 - val_accuracy: 0.7000
Epoch 9/10
63/63 [==============================] - 11s 182ms/step - loss: 0.5714 - accuracy: 0.7025 - val_loss: 0.5744 - val_accuracy: 0.6960
Epoch 10/10
63/63 [==============================] - 12s 184ms/step - loss: 0.5589 - accuracy: 0.7160 - val_loss: 0.5459 - val_accuracy: 0.7100
Training time: 118.70 seconds
63/63 [==============================] - 1s 16ms/step - loss: 0.5658 - accuracy: 0.7140
Test accuracy: 0.714

============================================================
TRAINING SUMMARY
============================================================
Device used: GPU
Epochs: 10

Base Model:
  Training time: 35.71 seconds
  Test accuracy: 0.708

Augmented Model:
  Training time: 118.70 seconds
  Test accuracy: 0.714

Improvement: 0.006
PS C:\Users\nagappans\ai_ml\ai_ml\NNProjects> 
