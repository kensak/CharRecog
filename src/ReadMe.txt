CharRecog のビルド方法
=====================

以下に、私が使用している環境と、CharRecog をビルドする手順について説明します。

環境
----
OS : Windows 7 64 ビット版
RAM : 8GB

必要なもの
---------
+ Micosoft Visual Studio Express 2012 for Windows Desktop  
  [ここ][MS]からダウンロードできます。

+ OpenCV 2.4.3  
  [ここ][OpenCV]からダウンロードできます。  
  ドキュメントに従って OpenCV をビルドしてください。  
  その際、設定により他のライブラリ（TBB など）をインストールする必要があります。  
  （付属の CharRecog プロジェクトでは、OpenCV はスタティック・ライブラリ、
  C/C++ のランタイムはダイナミック・リンクとしています。）
  
+ C++ AMP BLAS Library 1.0  
  [ここ][ampblas]からダウンロードできます。  
  
手順
----
1. `src/CharRecog.vcxproj` を Visual Studio で開きます。
1. プロジェクトのプロパティを開き、以下の項目を実際の環境に合わせて正しく設定します。
 - 構成プロパティ → VC++ ディレクトリ → インクルード　ディレクトリ
 - 構成プロパティ → VC++ ディレクトリ → ライブラリ　ディレクトリ
1. 構成（Release/Debug）とプラットフォーム(win32/x64)を正しくセットしてビルドしてください。

ヒント
----
+ double で計算をおこなう場合は、`config.h` で定義されている `REAL_IS_FLOAT` を
  未定義にしてください。
+ デフォルトではログが `log/log.txt` に書かれます。
  これを禁止するには `config.h` で定義されている `DEBUG_OUTPUT_LOG` を未定義にしてください。
  
榊原　研  
Email: ken.sakakibar@gmail.com
Blog: [http://kensak.github.io/](http://kensak.github.io/)  
GitHub: [https://github.com/kensak](https://github.com/kensak)  
Twitter: KenSakakibar

 
[MS]: http://www.microsoft.com/ja-jp/dev/express/
[OpenCV]: http://sourceforge.net/projects/opencvlibrary/files/opencv-win/
[ampblas]: http://ampblas.codeplex.com/releases/view/92383
