#NAME
CharRecog - ニューラルネットを使った文字認識エンジン

#SYNOPSIS
CharRecog []

#DESCRIPTION
プログラム CharRecog は

#OPTIONS
    "  -v        verbose mode\n"
    "  -L STR    layer param string to specify layer architecture,\n"
    "            which consists of layer param segment, concatenated by '_'.\n"
    "            Each segment is of format <type>,<param1>,<param2>,...\n"
    "            type: P for perceptron layer with tanh activation function, followed by\n"
    "                    inSize,outSize[,dropoutRatio]\n"
    "                  L for perceptron layer with identity activation function, followed by\n"
    "                    inSize,outSize[,dropoutRatio[,maxWeightNorm]]\n"
    "                  C for convolution layer with tanh activation function, followed by\n"
    "                    inMapH,inMapW,filterH,filterW,numInMaps,numOutMaps\n"
    "                  M for max-pool or maxout layer, followed by\n"
    "                    filterH,filterW\n"
    "                  S for softmax layer, followed by\n"
    "                    inOutSize\n"
    "  -m STR    training method and its parameters\n"
    "            standard backprop:\n"
    "              BPROP[,<initial learning rate>,<final learning rate>,<learning rate decay ratio>,\n"
    "                    <initial momentum>,<final momentum>,<momentum decay epoch>]\n"
    "            rprop (default):\n"
    "              RPROP[,dw0,dw_plus,dw_minus,dw_max,dw_min]\n"
    "  -f NUM    index of the first layer to train for TRAIN\n"
    "  -l NUM    index of the last layer to train for AE\n"
    "  -h NUM    height of input images\n"
    "  -w NUM    width of input images\n"
    "            If the actual size is different, the image will be resized.\n"
    "  -i NUM    number of iteration for training (default 100),\n"
    "            max number of samples to evaluate for testing (default INT_MAX)\n"
    "  -b        background of input images is black\n"
    "  -e STR    list of supported image file extensions.\n"
    "            Example: \"$JPG$JPEG$PNG$TIFF$\""
    "  -u        true answer is unknown when testing\n"
    "  -c        in the preprocessing, cut out the region containing the character\n"
    "  -a        on each epoch, apply random affine transformations to the input images\n"
    "  -p STR    name of the neural network parameter file (default: NN_params.bin)\n"
    "  -E NUM    when the command is TRAIN, calculate the recognition error rate every this epochs\n"
    "  -C        when the command is WIMAGE, output cumulative weights as images\n"

#詳細
ログは log/log.txt に書き込まれる。

#AUTHOR
榊原	研

#REPORTING BUGS
Report bugs to <ken.sakakibar@gmail.com>.

(END)