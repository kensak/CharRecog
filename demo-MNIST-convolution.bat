bin64\CharRecog.exe -b -h 28 -w 28 -L C,28,28,5,5,1,16,0.2_M,2,2_C,12,12,5,5,16,32,0.5_M,2,2_L,512,10_S,10 -B 6000 -i 200 -E 20 -m RPROP -p NN_conv.bin TRAIN data\MNIST\MNIST_train_data data\MNIST\MNIST_test_data
bin64\CharRecog.exe -p NN_conv.bin WIMAGE
pause
