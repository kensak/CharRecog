CharRecog
=========

ニューラルネットを使った文字認識エンジンです。
A character recognition engine using neural networks.


動作条件
--------
OS は Windows 7 またはそれ以上の 64 ビット版が必要です。
（ソースからは 32 ビット版をビルドすることもできます。）


まず動かしてみたい方へ（MNIST画像セットの学習）
----------------------------------------

1. GitHub からローカルにモジュール一式をダウンロードします。
```
git clone https://github.com/kensak/CharRecog.git
```

2. [THE MNIST DATABASE of handwritten digits のサイト](http://yann.lecun.com/exdb/mnist/)
から 4 つのデータファイルをダウンロードし、解凍してから data\MNIST フォルダーに置きます。

3. 同じフォルダーにある run.bat を実行すると、MNIST_test_data と MNIST_train_data というフォルダーができ、
png 形式の文字画像が書き込まれます。

4. ルートにある demo-MNIST.bat を実行すると学習が始まります。
200 ループを行い、10 ループごとに学習セットと評価セットでの認識率を計算します。
学習が終了すると、3 つのパーセプトロン・レイヤーの重み情報を画像にして出力します。
ネットワークの情報は NN_params.bin に出力され、文字認識に使用できます。  
例えば、
```
bin64\CharRecog.exe -v -b -h 28 -w 28 TEST data\MNIST\MNIST_test_data
```
を実行すると、評価セットの認識率を再び計算します。

 
パラメータの解説
----------------
リストの間に空行を挟むと、それぞれのリストに `<p>` タグが挿入され、行間が
広くなります。
 
    def MyFunction(param1, param2, ...)
 
+   `param1` :
    _パラメータ1_ の説明
 
+   `param2` :
    _パラメータ2_ の説明
 
関連情報
--------
### リンク、ネストしたリスト
1. [リンク1](http://example.com/ "リンクのタイトル")
    * ![画像1](http://github.com/unicorn.png "画像のタイトル")
2. [リンク2][link]
    - [![画像2][image]](https://github.com/)
 
  [link]: http://example.com/ "インデックス型のリンク"
  [image]: http://github.com/github.png "インデックス型の画像"
 
### 引用、ネストした引用
> これは引用です。
>
> > スペースを挟んで `>` を重ねると、引用の中で引用ができますが、
> > GitHubの場合、1行前に空の引用が無いと、正しくマークアップされません。
 
ライセンス
--------
Dual licensed under the [MIT license][MIT] and [GPL license][GPL].
[MIT]: http://www.opensource.org/licenses/mit-license.php
[GPL]: http://www.gnu.org/licenses/gpl.html