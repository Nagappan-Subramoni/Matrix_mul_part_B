PS C:\Users\nagappans\ai_ml\ai_ml\NNProjects> python .\catsvsdogs2.py --device cpu
Forced CPU usage
Using device: CPU
Loading datasets...
Found 2000 files belonging to 2 classes.
2025-06-07 17:27:27.255182: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Found 1000 files belonging to 2 classes.
Found 2000 files belonging to 2 classes.
Creating base model...

==================================================
Training Base Model
==================================================
Epoch 1/10
63/63 [==============================] - 24s 360ms/step - loss: 0.7669 - accuracy: 0.5115 - val_loss: 0.6905 - val_accuracy: 0.5090
Epoch 2/10
63/63 [==============================] - 24s 380ms/step - loss: 0.7480 - accuracy: 0.5515 - val_loss: 0.6774 - val_accuracy: 0.5480
Epoch 3/10
63/63 [==============================] - 25s 391ms/step - loss: 0.6784 - accuracy: 0.5950 - val_loss: 0.6625 - val_accuracy: 0.5960
Epoch 4/10
63/63 [==============================] - 25s 389ms/step - loss: 0.6500 - accuracy: 0.6250 - val_loss: 0.6613 - val_accuracy: 0.6360
Epoch 5/10
63/63 [==============================] - 26s 411ms/step - loss: 0.6262 - accuracy: 0.6745 - val_loss: 0.5995 - val_accuracy: 0.6580
Epoch 6/10
63/63 [==============================] - 26s 406ms/step - loss: 0.5645 - accuracy: 0.7220 - val_loss: 0.6127 - val_accuracy: 0.6740
Epoch 7/10
63/63 [==============================] - 31s 487ms/step - loss: 0.5188 - accuracy: 0.7420 - val_loss: 0.6776 - val_accuracy: 0.6840
Epoch 8/10
63/63 [==============================] - 31s 492ms/step - loss: 0.4881 - accuracy: 0.7670 - val_loss: 0.5752 - val_accuracy: 0.7200
Epoch 9/10
63/63 [==============================] - 39s 620ms/step - loss: 0.4429 - accuracy: 0.7955 - val_loss: 0.6302 - val_accuracy: 0.7060
Epoch 10/10
63/63 [==============================] - 35s 554ms/step - loss: 0.3934 - accuracy: 0.8195 - val_loss: 0.5537 - val_accuracy: 0.7290
Training time: 283.79 seconds
63/63 [==============================] - 6s 87ms/step - loss: 0.5637 - accuracy: 0.7295
Test accuracy: 0.729

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
63/63 [==============================] - 39s 577ms/step - loss: 0.7391 - accuracy: 0.5120 - val_loss: 1.1819 - val_accuracy: 0.5000
Epoch 2/10
63/63 [==============================] - 31s 491ms/step - loss: 0.7007 - accuracy: 0.5265 - val_loss: 0.6894 - val_accuracy: 0.5030
Epoch 3/10
63/63 [==============================] - 42s 667ms/step - loss: 0.7146 - accuracy: 0.5715 - val_loss: 0.6357 - val_accuracy: 0.6300
Epoch 4/10
63/63 [==============================] - 44s 692ms/step - loss: 0.6589 - accuracy: 0.6140 - val_loss: 0.6329 - val_accuracy: 0.6200
Epoch 5/10
63/63 [==============================] - 42s 670ms/step - loss: 0.6428 - accuracy: 0.6420 - val_loss: 0.5982 - val_accuracy: 0.6830
Epoch 6/10
63/63 [==============================] - 44s 697ms/step - loss: 0.6196 - accuracy: 0.6590 - val_loss: 0.6818 - val_accuracy: 0.5880
Epoch 7/10
63/63 [==============================] - 47s 743ms/step - loss: 0.6145 - accuracy: 0.6710 - val_loss: 0.6372 - val_accuracy: 0.6040
Epoch 8/10
63/63 [==============================] - 42s 673ms/step - loss: 0.5989 - accuracy: 0.6870 - val_loss: 0.6171 - val_accuracy: 0.6580
Epoch 9/10
63/63 [==============================] - 44s 697ms/step - loss: 0.5820 - accuracy: 0.7035 - val_loss: 0.5784 - val_accuracy: 0.6960
Epoch 10/10
63/63 [==============================] - 41s 649ms/step - loss: 0.5751 - accuracy: 0.7110 - val_loss: 0.5442 - val_accuracy: 0.7120
Training time: 415.92 seconds
63/63 [==============================] - 8s 120ms/step - loss: 0.5705 - accuracy: 0.7065
Test accuracy: 0.706

============================================================
TRAINING SUMMARY
============================================================
Device used: CPU
Epochs: 10

Base Model:
  Training time: 283.79 seconds
  Test accuracy: 0.729

Augmented Model:
  Training time: 415.92 seconds
  Test accuracy: 0.706

Improvement: -0.023
PS C:\Users\nagappans\ai_ml\ai_ml\NNProjects> python .\catsvsdogs2.py --device gpu
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

Conclusion: 
For CPU this is the timings:
============================================================
TRAINING SUMMARY
============================================================
Device used: CPU
Epochs: 10

Base Model:
  Training time: 283.79 seconds
  Test accuracy: 0.729

Augmented Model:
  Training time: 415.92 seconds
  Test accuracy: 0.706

for GPU this is the timings:
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
