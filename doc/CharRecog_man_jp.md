#名前
   CharRecog - ニューラルネットを使った文字認識エンジン

#書式
   CharRecog {options} <command> [<data-dir> [<data-dir2>]]

#説明
   プログラム CharRecog はニューラルネットを使った文字認識エンジンです。  
   <command> の値により以下の動作をおこないます。
   TRAIN     <data-dir> にある複数の画像ファイルを元に、それらの属するクラスを学習します。  
                画像フォーマットは　-e オプションに含まれる種別に対応します。  
				れぞれの画像ファイルの名前の一文字目 (2 バイト文字の場合はshift-JISコードの値)を  
				その文字のクラスとみなします。
				-p オプションで指定されるパラメターファイルがすでにあれば、そこからネットワークの
				パラメターを読み、さらに学習をおこないます。
				パラメターファイルがなければ -L オプションで指定される値からネットワークを構築し、
				ゼロから学習をおこないます。
   TEST      <data-dir> にある画像のクラスを予想します。
                -u オプションがある場合、それぞれの画像の予想クラスを表示します。
				-u オプションがない場合、ファイル名の一文字目 (2 バイト文字の場合はshift-JISコードの値)
				を その文字の真のクラスとみなし、予想されたクラスと一致した文字の割合（認識率）
				を表示します。-v オプションがあれば、それぞれの文字の予想クラスも表示します。
   AE          Autoencoding による学習をおこないます。その他の点は TRAIN の場合と同様です。
   WIMAGE  各レイヤーの重みを画像にして出力します。
				
   ログは log/log.txt に書き込まれます。

#オプション

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

#作者
榊原	研

#バグ報告
バグ報告は <ken.sakakibar@gmail.com> までお願いします。

(END)