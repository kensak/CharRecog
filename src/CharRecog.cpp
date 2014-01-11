/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

#include "stdafx.h"
#include "config.h"

#include <fstream>
#include <opencv2/opencv.hpp>
#include "sString.h"
#include "dict.h"
#include "CleanSmallIslands.h"
#include "NeuralNet.h"
#include "log.h"

// in Similitude.cpp
float MatchingResidue(const cv::Mat &templImage, const cv::Mat &sampleImage);

inline static bool SJIS1(const unsigned char c) { return ((0x81 <= c && c <= 0x9f) || (0xe0 <= c && c <= 0xfc)); }
inline static bool SJIS2(const unsigned char c) { return (0x40 <= c && c <= 0xfc); }

static void  PrintUsage(const char *exePath)
{
  printf_s("%s - Character recognition program, version of 2014/01/09\n", exePath);
  printf_s(
    "Usage: CharRecog {options}\n"
    "       <command> [<data-dir> [<data-dir2>]]\n\n"
    "command     TRAIN | TEST | AE | WIMAGE \n"
    "data-dir    train or test data directory\n\n"
    "data-dir2   when the command is TRAIN, specift the test data directory for evaluation\n\n"
    "Options\n"
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
    );
}

struct ParamType {
  std::string command;
  std::string dataDir;
  std::string dataDir2;
  bool verbose;
  int firstLayerToTrain;
  int lastLayerToTrain;
  int imageHeight;
  int imageWidth;
  int maxIter;
  int maxTestImage;
  bool blackBackground;
  std::string supportedExtensionList;
  std::string layerParamStr;
  bool trueAnswerUnknown;
  bool cutOutBlackRegion;
  bool randomAffineTransform;
  updateParam update_param;
  std::string paramFileName;
  int evaluateEvery;
  bool cumulative;

  ParamType() : verbose(false), firstLayerToTrain(0), lastLayerToTrain(-1),
    imageHeight(24), imageWidth(24), maxIter(100), maxTestImage(INT_MAX), blackBackground(false),
    trueAnswerUnknown(false), cutOutBlackRegion(false), randomAffineTransform(false),
    supportedExtensionList("$BMP$GIF$ICO$JPG$JPEG$PBM$PCD$PGM$PCT$PICT$PIC$PNG$PPM$PSD$TIF$TIFF$XBM$XPM$"),
    paramFileName("NN_params.bin"), evaluateEvery(0), cumulative(false)
  {}
  void log() const
  {
    Log("Program parameters read\n");
    Log("  command                    %s\n", command.c_str());
    Log("  NN parameter file          %s\n", paramFileName.c_str());
    Log("  image size                 %dx%d\n", imageHeight, imageWidth);
    Log("  background is black        %s\n", blackBackground? "true" : "false");
    Log("  supported extension list   %s\n", supportedExtensionList.c_str());
    Log("  cut out black region       %s\n", cutOutBlackRegion? "true" : "false");
    Log("  random affine transform    %s\n", randomAffineTransform? "true" : "false");

    if (command == "TRAIN")
    {
      Log("  training set dir           %s\n", dataDir.c_str());
      if (!dataDir2.empty())
      {
        Log("  test set dir               %s\n", dataDir2.c_str());
        Log("  max # of test images       %d\n", maxTestImage);
      }
      Log("  first layer to train       %d\n", firstLayerToTrain);
      Log("  max iteration              %d\n", maxIter);
      Log("  evaluate error every       %d\n", evaluateEvery);
    }
    else if (command == "AE")
    {
      Log("  training set dir           %s\n", dataDir.c_str());
      Log("  last layer to autoencode   %d\n", lastLayerToTrain);
    }
    else if (command == "TEST")
    {
      Log("  test set dir               %s\n", dataDir.c_str());
      Log("  max # of test images       %d\n", maxTestImage);
      Log("  true answer is unknown     %s\n", trueAnswerUnknown? "true" : "false");
    }
    else if (command == "WIMAGE")
    {
      Log("  output cumulative weights  %s\n", cumulative? "true" : "false");
    }

    if (update_param.type == updateParam::rprop)
    {
      Log("  training method          RPROP\n");
      Log("  dw0                        %f\n", update_param.dw0);
      Log("  dw_plus                    %f\n", update_param.dw_plus);
      Log("  dw_minus                   %f\n", update_param.dw_minus);
      Log("  dw_max                     %f\n", update_param.dw_max);
      Log("  dw_min                     %f\n", update_param.dw_min);
    }
    else if (update_param.type == updateParam::bprop)
    {
      Log("  training method          BPROP\n");
      Log("  initial learning rate      %f\n", update_param.learningRate);
      Log("  final learning rate        %f\n", update_param.finalLearningRate);
      Log("  learning rate decay rate   %f\n", update_param.learningRateDecay);
      Log("  initial momentum           %f\n", update_param.initMomentum);
      Log("  final momentum             %f\n", update_param.finalMomentum);
      Log("  momentum decay epoch       %d\n", update_param.momentumDecayEpoch);
    }
  }
} param;

// 白い余白を追加して正方形にする。
// img は 8bit 白黒画像と仮定。
void Square(cv::Mat &img)
{
	if (img.rows == img.cols)
		return;

	if (img.rows > img.cols)
	{
		int diff= img.rows - img.cols;
		cv::copyMakeBorder(img, img, 0, 0, diff / 2, diff - diff / 2, cv::BORDER_CONSTANT, 255);
	}
	else
	{
		int diff = img.cols - img.rows;
		cv::copyMakeBorder(img, img, diff / 2, diff - diff / 2, 0, 0, cv::BORDER_CONSTANT, 255);
	}
}

static void CutOutBlackRegion(cv::Mat &image, const int threshold)
{
	int top = -1, bottom = -1, left = -1, right = -1;

	for (int y = 0; y < image.rows && top == -1; ++y)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			if (image.at<uchar>(y, x) <= threshold)
			{
				top = y;
				break;
			}
		}
	}

	if (top == -1)
		return;

	for (int y = image.rows - 1; y >= 0 && bottom == -1; --y)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			if (image.at<uchar>(y, x) <= threshold)
			{
				bottom = y;
				break;
			}
		}
	}

	for (int x = 0; x < image.cols && left == -1; ++x)
	{
		for (int y = 0; y < image.rows; ++y)
		{
			if (image.at<uchar>(y, x) <= threshold)
			{
				left = x;
				break;
			}
		}
	}

	for (int x = image.cols - 1; x >= 0 && right == -1; --x)
	{
		for (int y = 0; y < image.rows; ++y)
		{
			if (image.at<uchar>(y, x) <= threshold)
			{
				right = x;
				break;
			}
		}
	}

	cv::Mat tmpImg(image, cv::Rect(left, top, right - left + 1, bottom - top + 1));
	image = tmpImg;
}

void CoerceAverageBrightness(cv::Mat &image, const int avr)
{
	int cols = image.cols, rows = image.rows;
	if (image.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}

	int lastSum = -1;
	while (true)
	{
		int count = 0;
		int sum = 0;

		for (int i = 0; i < rows; ++i)
		{
			const unsigned char* p = image.ptr<unsigned char>(i);
			for(int j = 0; j < cols; j++)
			{
				if (p[j])
				{
					++count;
					sum += p[j];
				}
			}
		}

    if (sum == lastSum)
      return;
		lastSum = sum;

		int currentAvr = (sum + (count / 2)) / count;

		if (currentAvr == avr)
			return;

		image *= (double(avr) / currentAvr);
		
		if (currentAvr > avr)
			return;
	}
}

// ファイルから読み込んだ画像の形を整えて image に返す。
static int GetNormalizedImage(
  cv::Mat &image,
  const char *filename,
  const int height,
  const int width,
  const int cutOutThreshold)
{
	image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

	if (image.empty())
		return 1;

	// 地を黒から白へ反転する。
  if (param.blackBackground)
	  image = 255 - image;

	// 白い余白を追加して正方形にする。
	Square(image);

  if (param.cutOutBlackRegion)
  {
	  // 大きさを 64x64 にそろえてから小さなゴミを除去する。
	  if (image.rows != 64 || image.cols != 64)
      cv::resize(image, image, cv::Size(64, 64), 0, 0, CV_INTER_AREA);
    const int minNumBlack = 20;
    const int lum1 = 220;
    const int lum2 = 130;
	  CleanSmallIslands(image, image, lum1, lum2, minNumBlack, false);

	  // 黒い文字の部分だけを切り出す
	  CutOutBlackRegion(image, cutOutThreshold);

	  // 白い余白を追加して正方形にする。
	  Square(image);
  }

	// 大きさをそろえる。
	if (image.rows != height || image.cols != width)
		cv::resize(image, image, cv::Size(height, width), 0, 0, CV_INTER_AREA);

	// 地を白から黒へ反転する。
  image = 255 - image;

	// 明るさが 0 でない pixel value の平均が与えられた数値になるよう
	// image の明るさを調整する。
	CoerceAverageBrightness(image, 170);

	/*
	// ピクセル値の範囲が [0, 255] となるように調整する。
	double minVal, maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	image = (image - (uchar)minVal) * (255 / (maxVal - minVal));
	*/

#if 0 // チェック用：加工後の画像をファイルに落とす。
	sjisString fname("forCheck\\");
	const char *lastPosBackSlash = strrchr(filename, '\\');
	if (lastPosBackSlash != NULL)
		fname += lastPosBackSlash + 1;
	else
		fname += filename;
	cv::imwrite(fname.pData(), image);
#endif

	return 0;
}

class FeaturePixelValues
{
public:
	FeaturePixelValues() {}
	int appendTo(cv::Mat &featureData, const cv::Mat &image)
	{
    if (image.total() != featureData.cols)
    {
      Log("Error: image size %dx%d=%d does not match the feature dimension %d.\n", image.rows, image.cols, image.total(), featureData.cols);
      return 1;
    }

    // [0, 255] を [-1.0, 1.0) にマップする。
		cv::Mat dst;
		image.convertTo(dst, CV_REAL, 1.0 / 127.5, -1.0);

    featureData.push_back(dst.reshape(0, 1));

    return 0;
	}
};

FeaturePixelValues fpv;

static inline char *charStr(const unsigned _int16 c)
{
  static char str[3];
  if (SJIS1(c >> 8))
  {
    str[0] = c >> 8;
    str[1] = c & 0xff;
    str[2] = 0;
  }
  else
  {
    str[0] = c & 0xff;
    str[1] = 0;
  }
  return str;
}

static inline unsigned _int16 swap(const unsigned _int16 x)
{
	return ((x >> 8) | ((x & 0xff) << 8));
}

static inline unsigned _int16 strChar(const char *str)
{
  if (SJIS1(str[0]))
    return swap(*reinterpret_cast<const unsigned _int16 *>(str));
  return (unsigned char)str[0];
}

static int AppendAllFeatures(
  cv::Mat &featureData,
  unsigned _int16 &charCode,
  const char *folderName,
  const char *fileName)
{
	sjisString file = sjisString(folderName) + fileName;

	cv::Mat image;
  GetNormalizedImage(image, file.pData(), param.imageHeight, param.imageWidth, 150);

#if 0
	// 整形した画像をファイルに書き出す。
	file = sjisString("tmp\\") + fileName;
	cv::imwrite(file.pData(), image);
#endif

  // ここで値を [-1.0, 1.0) に正規化している。
	if (fpv.appendTo(featureData, image) != 0)
    return 1;

	charCode = strChar(fileName);

  return 0;
}

static int Predict(
	const char *samplesFolder_,
	const int testDataMax,
  NeuralNet &nn
	)
{
	int numTestData = 0, numGood = 0;

	sjisString samplesFolder(samplesFolder_);
  	if (samplesFolder[samplesFolder.length() - 1] != '\\')
		samplesFolder += '\\';
	sjisString wildcard = samplesFolder + "*"; 

  cv::Mat samples(0, param.imageHeight * param.imageWidth, CV_REAL);
  std::vector<unsigned _int16> trueChar;

	WIN32_FIND_DATAA fileData;
	HANDLE hdl = FindFirstFileA(wildcard.pData(), &fileData);
	if ( hdl != INVALID_HANDLE_VALUE )
	{
		sjisString extension;

		do
		{
			if( (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 )
			{
				static char supportedExtensionList[] = "$BMP$GIF$ICO$JPG$JPEG$PBM$PCD$PGM$PCT$PICT$PIC$PNG$PPM$PSD$TIF$TIFF$XBM$XPM$";
				extension = "$" + GetExtension(fileData.cFileName) + "$";
				extension.toUpper();

				if (strstr(supportedExtensionList, extension.pData()) != NULL)
				{
					unsigned _int16 c;
          int ret = AppendAllFeatures(samples, c, samplesFolder.pData(), fileData.cFileName);
          if (ret != 0)
          {
						Log("Error: AppendAllFeatures() returned %d.\n", ret);
            return 1;
          }

          trueChar.push_back(c);
					++numTestData;

					if( numTestData > testDataMax )
						break;
				}
			}
		} while( FindNextFileA(hdl, &fileData) );
		FindClose( hdl );
	}

	// それぞれの文字である確率のベクトルが mlp_response に得られる。
	cv::Mat mlp_response;
  int ret = nn.predict(samples, mlp_response);
  if (ret != 0)
  {
		Log("Error: NeuralNet::predict() returned %d.\n", ret);
    return 2;
  }

	// 各行を確率の高い順にソートする。
	cv::Mat dst;
	cv::sortIdx(mlp_response, dst, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);

  for (int s = 0; s < mlp_response.rows; ++s)
  {
    unsigned _int16 predictedChar = (unsigned _int16)((unsigned _int16)dst.at<int>(s, 0) + nn.firstCharCode);

    if (param.trueAnswerUnknown)
    {
			Log("%s", charStr(predictedChar));
			printf("%s", charStr(predictedChar));
    }
    else
    {
#if 0 // 5つの候補とテンプレート画像を比べる。

			// 現在の文字画像を sampleImage に得る。
			sjisString file = samplesFolder + fileData.cFileName;
			cv::Mat sampleImage;
			GetNormalizedImage(sampleImage, file.pData(), param.imageHeight, param.imageWidth, 150);

			// 上位をいくつか表示する。
			int pos;
			cv::Mat templateImage;
			float residue = FLT_MAX;
			unsigned _int16 c3;
			for(int i = 0; i < 5; ++i)
			{
				pos = dst.at<int>(s, i);
				unsigned _int16 c2 = (unsigned _int16)pos + firstCharCode;
				// fprintf(fpOut, "%s(%.3lf%%) ", charStr(c2), 100.0 * mlp_response.at<double>(0, pos));

				for (int j = 1; ; ++j)
				{
					// 上位の候補の文字画像を templateImage に得る。
					sjisString templFileName = format("template\\%s%d.png", charStr(c2), j);

					if (_access(templFileName.pData(), 0) == -1)
						break;

					templateImage = cv::imread(templFileName.pData(), cv::IMREAD_GRAYSCALE);

					// テンプレート画像とサンプル画像を比べる。
          Log("Matching %s: ", templFileName.pData());
					float newResidue = MatchingResidue(templateImage, sampleImage);
					if (newResidue < residue)
					{
						residue = newResidue;
						c3 = c2;
					}
				}
			}
						
			//fprintf(fpOut, "%s -> %s (%s?) (%s)\n", charStr(trueChar), charStr(predictedChar), charStr(c3), fileData.cFileName);

			if (trueChar == c3)
			{
				Log("%s(%24s) OK\n", charStr(trueChar), fileData.cFileName);
				++numGood;
			}
			else
			{
				int origChar = trueChar - firstCharCode;
				int order = -1;
				int *p = dst.ptr<int>(s); 
				for (int i = 0; i < dst.cols; ++i)
				{
					if (p[i] == origChar)
					{
						order = i + 1;
						break;
					}
				}

				Log("%s(%24s) %5dth %s", charStr(trueChar), fileData.cFileName, order, charStr(c3));
				for(int i = 0; i < 5; ++i)
				{
					pos = dst.at<int>(s, i);
					unsigned _int16 c2 = (unsigned _int16)pos + firstCharCode;
					Log(" %s(%.2lf%%)", charStr(c2), 100.0 * mlp_response.at<double>(0, pos));
				}
				Log("\n");
			}

#else // テンプレート画像と比較しない。

			if (trueChar[s] == predictedChar)
			{
        if (param.verbose)
        {
  				Log("%s(%24s) OK\n", charStr(trueChar[s]), fileData.cFileName);
  				printf("%s(%24s) OK\n", charStr(trueChar[s]), fileData.cFileName);
        }
				++numGood;
			}
      else if (param.verbose)
			{
				int origChar = trueChar[s] - nn.firstCharCode;
				int order = -1;
				int *p = dst.ptr<int>(s); 
				for (int i = 0; i < dst.cols; ++i)
				{
					if (p[i] == origChar)
					{
						order = i + 1;
						break;
					}
				}

				Log("%s(%24s) %5dth ", charStr(trueChar[s]), fileData.cFileName, order);
				printf("%s(%24s) %5dth ", charStr(trueChar[s]), fileData.cFileName, order);
				for(int i = 0; i < 5; ++i)
				{
					int pos = dst.at<int>(s, i);
					unsigned _int16 c2 = (unsigned _int16)pos + nn.firstCharCode;
					Log(" %s(%.2f%%)", charStr(c2), 100.0 * mlp_response.at<real>(0, pos));
					printf(" %s(%.2f%%)", charStr(c2), 100.0 * mlp_response.at<real>(0, pos));
				}
				Log("\n");
				printf("\n");
			}
#endif
    }
  }

  if (param.verbose)
  {
		Log("\n");
		printf("\n");
  }

  if (!param.trueAnswerUnknown)
  {
	  // 正答率の表示
	  Log("Recognition rate: %.3f%% (%d / %d)\n", 100.0f * numGood / numTestData, numGood, numTestData);
	  printf("Recognition rate: %.3f%% (%d / %d)\n", 100.0f * numGood / numTestData, numGood, numTestData);
  }

	return 0;
}

void MakeSjisKanjiList(int lineLen)
{
	FILE *fp;
    fopen_s(&fp, "kanji.txt", "wt");

	int count = 0;
	for(unsigned int c = 0x8141; c <= 0xeeec; ++c)
	{
		if (SJIS1(c >> 8) && SJIS2(c & 0xff))
		{
			fprintf(fp, "%s", charStr(c));
			++count;
			if ((count % lineLen) == 0)
				fprintf(fp, "\n");
		}
	}

	fclose(fp);
}

// 標準偏差が stddev のガウスノイズを加えたあと、
// -stddev が 0 に、255 + stddev が 255 になるよう調整する。
static void AddGaussianNoise(cv::Mat &image, const float stddev)
{
	image.convertTo(image, CV_32FC1);
	cv::Mat noise(image.size(), CV_32FC1);
	randn(noise, std::vector<int>(1, 0), std::vector<float>(1, stddev));
	image += noise;
	double alpha = 255.0 / (255 + stddev * 2);
	image.convertTo(image, CV_8UC1, alpha, stddev * alpha);
}

static void CreateTransformedSamples(const char *folder)
{
	sjisString wildcard = folder;
	wildcard += "\\*";

	WIN32_FIND_DATAA fileData;
	HANDLE hdl = FindFirstFileA(wildcard.pData(), &fileData);
    if ( hdl == INVALID_HANDLE_VALUE )   // no file in the directory
	{
		// Do nothing.
		return;
	}
	else
	{
		const double d = 1.0 / 16.0;
		std::vector<cv::Mat> matrices;
		matrices.push_back((cv::Mat_<double>(2,3) << 1-d,  d, 0,  0,   1, 0)); // x方向 せん断 +
		matrices.push_back((cv::Mat_<double>(2,3) << 1-d, -d, d,  0,   1, 0)); // x方向 せん断 -
		matrices.push_back((cv::Mat_<double>(2,3) << 1,    0, 0,  d, 1-d, 0)); // y方向 せん断 +
		matrices.push_back((cv::Mat_<double>(2,3) << 1,    0, 0, -d, 1-d, d)); // y方向 せん断 -
		matrices.push_back((cv::Mat_<double>(2,3) << 1-d,  d, 0,  d, 1-d, 0)); // x, y方向 せん断 +
		matrices.push_back((cv::Mat_<double>(2,3) << 1-d, -d, d, -d, 1-d, d)); // x, y方向 せん断 -
		matrices.push_back((cv::Mat_<double>(2,3) << 1-d, -d, d,  d, 1-d, 0)); // 回転 -
		matrices.push_back((cv::Mat_<double>(2,3) << 1-d,  d, 0, -d, 1-d, d)); // 回転 +

		sjisString extension;
		do
		{
			if( (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 )
			{
				static char supportedExtensionList[] = "$BMP$GIF$ICO$JPG$JPEG$PBM$PCD$PGM$PCT$PICT$PIC$PNG$PPM$PSD$TIF$TIFF$XBM$XPM$";

				extension = "$" + GetExtension(fileData.cFileName) + "$";
				extension.toUpper();

				if (strstr(supportedExtensionList, extension.pData()) != NULL)
				{
					sjisString file = sjisString(folder) + "\\" + fileData.cFileName;
					cv::Mat image = cv::imread(file.pData(), cv::IMREAD_GRAYSCALE);

					cv::Mat outImage;
					for (int i = 0; i < (int)matrices.size(); ++i)
					{
						cv::Mat M;
						matrices[i].copyTo(M);
						M.at<double>(0, 2) *= image.cols;
						M.at<double>(1, 2) *= image.rows;
						cv::warpAffine(image, outImage, M, image.size(), cv::INTER_AREA /*+ cv::WARP_INVERSE_MAP*/, cv::BORDER_CONSTANT, 255);
						sjisString suffix = format("-m%d", i + 1);
						cv::imwrite(AppendToFileName(file, suffix.pData()).pData(), outImage);
					}
				}
			}
		}
		while( FindNextFileA(hdl, &fileData) );
		FindClose( hdl );
	}
}

static void RandomAffineTransform(cv::Mat &inputs, void *info)
{
  const int &sampleHeight = *(int *)info;
  const int numSamples = inputs.rows;
  const int sampleSize = inputs.cols;
  const int sampleWidth = sampleSize / sampleHeight;

	cv::RNG rng(time(NULL));
  real d = rng.uniform(real(0), real(1.0 / 8.0));
  cv::Mat M;
  switch (rng(8))
  {
  case 0: M = (cv::Mat_<real>(2, 3) << 1-d,  d, 0,                0,   1, 0               ); break; // x方向 せん断 +
	case 1: M = (cv::Mat_<real>(2, 3) << 1-d, -d, d * sampleWidth,  0,   1, 0               ); break; // x方向 せん断 -
	case 2:	M = (cv::Mat_<real>(2, 3) << 1,    0, 0,                d, 1-d, 0               ); break; // y方向 せん断 +
	case 3:	M = (cv::Mat_<real>(2, 3) << 1,    0, 0,               -d, 1-d, d * sampleHeight); break; // y方向 せん断 -
	case 4:	M = (cv::Mat_<real>(2, 3) << 1-d,  d, 0,                d, 1-d, 0               ); break; // x, y方向 せん断 +
	case 5:	M = (cv::Mat_<real>(2, 3) << 1-d, -d, d * sampleWidth, -d, 1-d, d * sampleHeight); break; // x, y方向 せん断 -
	case 6:	M = (cv::Mat_<real>(2, 3) << 1-d, -d, d * sampleWidth,  d, 1-d, 0               ); break; // 回転 -
	case 7:	M = (cv::Mat_<real>(2, 3) << 1-d,  d, 0,               -d, 1-d, d * sampleHeight); break; // 回転 +
  }

  for (int s = 0; s < numSamples; ++s)
  {
    cv::Mat image(sampleHeight, sampleWidth, inputs.type(), inputs.data + inputs.step[0] * s);
    cv::Mat outImage;
		cv::warpAffine(image, outImage, M, image.size(), cv::INTER_AREA /*+ cv::WARP_INVERSE_MAP*/, cv::BORDER_CONSTANT, -1);
    outImage.copyTo(image);
  }
}

static void NormalizeTemplates(const char *folder)
{
	sjisString wildcard = folder;
	wildcard += "\\seed\\*";

	WIN32_FIND_DATAA fileData;
	HANDLE hdl = FindFirstFileA(wildcard.pData(), &fileData);
    if ( hdl == INVALID_HANDLE_VALUE )   // no file in the directory
	{
		// Do nothing.
		return;
	}
	else
	{
		sjisString extension;
		do
		{
			if( (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 )
			{
				static char supportedExtensionList[] = "$BMP$GIF$ICO$JPG$JPEG$PBM$PCD$PGM$PCT$PICT$PIC$PNG$PPM$PSD$TIF$TIFF$XBM$XPM$";

				extension = "$" + GetExtension(fileData.cFileName) + "$";
				extension.toUpper();

				if (strstr(supportedExtensionList, extension.pData()) != NULL)
				{
					sjisString inFile = sjisString(folder) + "\\seed\\" + fileData.cFileName;
					cv::Mat image;
					GetNormalizedImage(image, inFile.pData(), 64, 64, 150);

					/* GetNormalizedImage() の中に移動。

					// 明るさが 0 でない pixel value の平均が与えられた数値になるよう
					// image の明るさを調整する。
					CoerceAverageBrightness(image, 170);

					*/

					unsigned _int16 trueChar = strChar(fileData.cFileName);
					sjisString outFile;
					for (int i = 1; ; ++i)
					{
						outFile = format("%s\\%s%d.png", folder, charStr(trueChar), i);
						if (_access(outFile.pData(), 0) == -1)
							break;
					}
					cv::imwrite(outFile.pData(), image);
				}
			}
		}
		while( FindNextFileA(hdl, &fileData) );
		FindClose( hdl );
	}
}

static int CountFilesInDir(
  const std::string &dataDir,
  const std::string &supportedExtensionList,
  unsigned _int16 &firstCharCode,
  int &class_count)
{
	std::string wildcard = dataDir + "*";
	int numFiles = 0;
  unsigned _int16 lastCharCode = 0;
  firstCharCode = USHRT_MAX;

	WIN32_FIND_DATAA fileData;
	HANDLE hdl = FindFirstFileA(wildcard.c_str(), &fileData);
  if (hdl != INVALID_HANDLE_VALUE)
	{
		sjisString extension;
    unsigned _int16 charCode;
		do
		{
			if ((fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0)
			{
				extension = "$" + GetExtension(fileData.cFileName) + "$";
				extension.toUpper();

				if (strstr(supportedExtensionList.c_str(), extension.pData()) != NULL)
        {
					++numFiles;
          if (SJIS1(fileData.cFileName[0]))
  					charCode = strChar(fileData.cFileName);
          else
            charCode = fileData.cFileName[0];

          if (charCode < firstCharCode)
            firstCharCode = charCode;
          if (lastCharCode < charCode)
            lastCharCode = charCode;
        }
			}
		}
		while( FindNextFileA(hdl, &fileData) );
        
		FindClose( hdl );
	}

  class_count = lastCharCode - firstCharCode + 1;

  return numFiles;
}

static int BuildNN(NeuralNet &nn, const char *paramFileName, bool readWeightFile = false)
{
  if (_access(paramFileName, 0) != -1)
  {
  	// ファイルから重みをロードする。
    nn.readBinary(paramFileName);
    Log("NN parameters has been read from file.\n");

    if (!param.layerParamStr.empty())
      Log("Layer parameters option has been ignored.\n");
  }
  else if (readWeightFile)
  {
    Log("Error: cannot find NN parameter file: %s\n", paramFileName);
    return 1;
  }
  else
  {
    /*
    // preceptron : 24x24 -> 256 -> 256 -> class_count
    layers.resize(3);
    layers[0].createPerceptronLayer(trainData.cols, 256);
    layers[1].createPerceptronLayer(256, 256);
    layers[2].createPerceptronLayer(256, class_count);

    // 24x24 --(5x5)-> 10@20x20 -> 10@10x10 --(5x5)-> 20@6x6 -> 20@3x3 -> 128 -> class_count
    layers.resize(6);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 5, 5), 1, 10);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[2].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 10, 10), (cv::Mat_<_int32>(1,2) << 5, 5), 10, 20);
    layers[3].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[4].createPerceptronLayer(180, 128);
    layers[5].createPerceptronLayer(128, class_count);

    // 24x24 --(5x5)-> 10@20x20 -> 10@10x10 --(5x5)-> 20@6x6 -> 256 -> class_count
    layers.resize(5);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 5, 5), 1, 10);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[2].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 10, 10), (cv::Mat_<_int32>(1,2) << 5, 5), 10, 20);
    layers[3].createPerceptronLayer(720, 256);
    layers[4].createPerceptronLayer(256, class_count);

    // 24x24 --(5x5)-> 20@20x20 -> 20@10x10 --(5x5)-> 20@6x6 -> 256 -> class_count
    layers.resize(5);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 5, 5), 1, 20);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[2].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 10, 10), (cv::Mat_<_int32>(1,2) << 5, 5), 20, 20);
    layers[3].createPerceptronLayer(720, 256);
    layers[4].createPerceptronLayer(256, class_count);

    // 24x24 --(5x5)-> 20@20x20 -> 20@10x10 --(5x5)-> 40@6x6 -> 40@3x3 -> 256 -> class_count
    layers.resize(6);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 5, 5), 1, 20);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[2].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 10, 10), (cv::Mat_<_int32>(1,2) << 5, 5), 20, 40);
    layers[3].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[4].createPerceptronLayer(360, 256);
    layers[5].createPerceptronLayer(256, class_count);

    // 24x24 --(5x5)-> 40@20x20 -> 40@10x10 --(5x5)-> 40@6x6 -> 40@3x3 -> 256 -> class_count
    layers.resize(6);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 5, 5), 1, 40);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[2].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 10, 10), (cv::Mat_<_int32>(1,2) << 5, 5), 40, 40);
    layers[3].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[4].createPerceptronLayer(360, 256);
    layers[5].createPerceptronLayer(256, class_count);

    // 24x24 --(7x7)-> 20@18x18 -> 20@6x6 -> 256 -> class_count
    layers.resize(4);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 7, 7), 1, 20);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 3, 3));
    layers[2].createPerceptronLayer(720, 256);
    layers[3].createPerceptronLayer(256, class_count);

    // 24x24 --(7x7)-> 40@18x18 -> 40@6x6 -> 256 -> class_count
    layers.resize(4);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 7, 7), 1, 40);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 3, 3));
    layers[2].createPerceptronLayer(1440, 256);
    layers[3].createPerceptronLayer(256, class_count);

    // 24x24 --(5x5)-> 20@20x20 -> 20@10x10 --(3x3)-> 20@8x8 -> 20@4x4 -> 256 -> class_count
    layers.resize(6);
    layers[0].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 24, 24), (cv::Mat_<_int32>(1,2) << 5, 5), 1, 20);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[2].createConvolutionLayer((cv::Mat_<_int32>(1,2) << 10, 10), (cv::Mat_<_int32>(1,2) << 3, 3), 20, 20);
    layers[3].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 2, 2));
    layers[4].createPerceptronLayer(320, 256);
    layers[5].createPerceptronLayer(256, class_count);

    // preceptron + softmax : 24x24 -> 256 -> 256 -(linear)-> class_count -> softmax layer
    layers.resize(4);
    layers[0].createPerceptronLayer(trainData.cols, 256);
    layers[1].createPerceptronLayer(256, 256);
    layers[2].createLinearPerceptronLayer(256, class_count);
    layers[3].createSoftMaxLayer(class_count);

    // maxPool + softmax : 24x24 -(linear)-> 256 -(maxPool 1x2)-> 128 -(linear)-> 128 -(maxPool 1x2)-> 64
    //                     -(linear)-> class_count -> softmax layer
    layers.resize(6);
    layers[0].createLinearPerceptronLayer(trainData.cols, 256, 0.2);
    layers[1].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 1, 2));
    layers[2].createLinearPerceptronLayer(128, 256, 0);
    layers[3].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << 1, 2));
    layers[4].createLinearPerceptronLayer(128, class_count, 0);
    layers[5].createSoftMaxLayer(class_count);
    */

    int ret = nn.create(param.layerParamStr);
    if (ret != 0)
    {
			Log("Error in NeuralNet::create() : return code = %d\n", ret);
      return 100 + ret;
    }
    Log("NN has been constructed from the layer param string.\n");
  }

  // log NN settings
  nn.logSettings();

  return 0;
}

static int ReadTrainData(
  cv::Mat &trainData,
  cv::Mat &responses,
  cv::Mat &sampleWeights,
  const std::string &dataDir,
  const std::string &supportedExtensionList,
  const int numData,
  const unsigned _int16 firstCharCode,
  const int class_count)
{
	Log("Reading training data...\n");

  trainData.create(0, param.imageHeight * param.imageWidth, CV_REAL);
  trainData.reserve(numData);
  responses.create(0, 1, CV_32S);
  responses.reserve(numData);

  // 同じ文字のサンプルがいくつあるかを記憶する。
  cv::Mat freq(1, class_count, CV_32S, cv::Scalar(0));

  std::string wildcard = param.dataDir + "*";
	WIN32_FIND_DATAA fileData;
	HANDLE hdl = FindFirstFileA(wildcard.c_str(), &fileData);
  if (hdl != INVALID_HANDLE_VALUE)
	{
    unsigned _int16 charCode;
		sjisString extension;
		do
		{
			if( (fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 )
			{
				extension = "$" + GetExtension(fileData.cFileName) + "$";
				extension.toUpper();

				if (strstr(param.supportedExtensionList.c_str(), extension.pData()) != NULL)
				{
          int ret = AppendAllFeatures(trainData, charCode, param.dataDir.c_str(), fileData.cFileName);
          if (ret != 0)
          {
						Log("Error: AppendAllFeatures() returned %d.\n", ret);
            return 1;
          }

					responses.push_back((_int32)charCode);
          ++(freq.at<_int32>(0, (_int32)(charCode - firstCharCode)));
				}
			}
		}
		while( FindNextFileA(hdl, &fileData) );
       
		FindClose( hdl );
	}

  // set sampleWeights
  sampleWeights.create(trainData.rows, 1, CV_REAL);
	for (int i = 0; i < trainData.rows; ++i)
	{
		int cls_label = (unsigned _int32)responses.at<_int32>(i, 0) - firstCharCode;
    sampleWeights.at<real>(i, 0) = real(1) / freq.at<_int32>(0, cls_label);
  }

  return 0;
}

static void CalculateRecognitionError(NeuralNet &nn)
{
  if (!param.dataDir.empty())
  {
	  Log("Calculating training set recognition rate...\n");
	  printf("Calculating training set recognition rate...\n");
    Predict(param.dataDir.c_str(), param.maxTestImage, nn);
  }

  if (!param.dataDir2.empty())
  {
  	Log("Calculating test set recognition rate...\n");
	  printf("Calculating test set recognition rate...\n");
    Predict(param.dataDir2.c_str(), param.maxTestImage, nn);
  }
}

/*
// for debugging of matMul()
void matMul(
  const cv::Mat &src1,
  const cv::Mat &src2,
  real alpha,
  cv::Mat &dst,
  int flags = 0);
void printMat(const cv::Mat &m);
*/

int _tmain(int argc, _TCHAR* argv[])
{
#if 0 // for debugging of matMul()
  NeuralNet nn2;
  float a[10*256] = {
-0.794208,-0.581803,-0.822304,0.771949,0.274785,-0.377922,-0.477354,-0.831960,0.048630,-0.663919,0.395677,0.050741,-0.634959,0.483813,-0.591699,1.056844,0.415493,0.079895,-0.230756,-0.733630,0.371766,0.573485,-0.313121,0.403744,-0.086830,0.862685,0.514275,-0.087365,0.292781,-0.451492,-0.761814,0.677791,0.859239,0.300853,0.369943,-0.397329,-0.816134,-0.359555,-0.935626,0.887233,-0.738867,0.516509,-0.858654,-0.647332,-0.208832,-0.789105,0.188270,-0.163384,-0.468486,0.439837,-0.616638,-0.448807,-0.567376,-0.736429,0.828085,0.913395,-0.366336,-0.844765,0.664725,-0.407981,-0.333249,-0.818073,0.851085,-0.941443,-0.006690,-1.094275,-0.452697,0.822902,0.896872,-0.555290,-0.077008,0.242093,0.743022,-1.076466,-0.656912,0.897107,0.748904,-0.856278,-0.848148,-0.839123,-0.406387,-0.920253,0.061111,-0.810088,0.625560,-0.724146,0.655787,-0.474114,-0.969453,-0.109353,0.796942,-0.162231,-0.342369,-0.229867,-1.081612,0.235364,-0.383188,-0.751644,0.642879,-0.475179,-0.005985,-0.413469,0.687666,0.914683,0.454065,-0.881049,-0.515854,0.726129,0.504197,0.578119,0.615250,0.845056,-1.105032,-0.937167,0.239367,-0.255697,0.682025,-1.102003,0.846228,0.871262,0.222854,0.545337,-0.849292,0.763525,0.260834,-0.131103,0.373753,-1.077126,0.687051,-0.194059,-0.372340,0.526880,0.934737,0.093190,0.481682,-1.039682,0.503077,-0.296978,-0.754811,0.298682,-0.438823,-0.678078,0.919860,-0.461664,0.779633,0.003989,1.007832,-0.637537,-0.970118,0.272778,0.416809,0.036095,0.767234,-0.387882,-0.692792,0.014016,0.830616,-0.389738,-0.534663,0.373212,-0.974198,-0.111253,0.341418,0.160824,-0.239764,0.280098,0.895276,-0.226393,0.722607,0.892204,0.837611,0.880168,-0.857092,-1.071570,-0.502604,-0.460081,-0.829686,0.791101,0.870662,-0.806021,-0.138382,-0.659866,-0.901237,0.434635,-0.581179,0.079217,-0.408555,-0.735167,0.355555,0.288707,-0.140690,0.770476,-0.641227,0.159898,-0.444133,-0.743286,-0.147062,0.361930,0.347728,-0.389070,-0.559550,-0.539702,0.681977,0.357084,-0.051111,-0.108373,0.191696,-0.385503,0.321663,-0.045673,-0.619887,0.818396,-0.324242,0.295087,-0.900594,0.449312,-0.662832,-0.220339,-0.311212,-0.642539,0.022824,-0.826499,0.214055,-0.611664,0.438912,-0.347216,0.569596,-0.423790,-0.391046,-0.072928,-0.322947,-0.412585,-0.178668,-0.093134,-0.217153,0.230123,0.473020,0.499059,-0.578180,0.139918,0.103356,0.074693,0.696702,0.429175,-0.918330,-0.771421,0.595502,0.031837,-0.695373,-0.192882,0.462349,0.273723,0.780863,-0.107222,-0.240055,-0.850410,
-0.473739,0.239271,-0.612249,0.331030,-0.663632,0.691014,-0.482232,-0.805622,-0.245483,0.115769,0.856090,-0.202040,0.273738,0.723606,0.223305,0.176023,0.439170,-0.326379,-0.445549,-0.049256,-0.593117,0.397645,-0.146329,-0.293162,-0.651694,0.438788,-0.338716,0.047866,-0.242338,-0.020884,-0.319167,-0.001843,0.081356,0.116133,-0.240755,-0.984273,0.008334,-0.592804,0.062566,0.740811,-0.538505,-0.185491,-0.027885,0.545960,0.474153,-0.100056,-0.037597,0.461494,0.271855,0.781036,0.024304,-0.675867,0.152273,0.159630,0.044320,0.451284,0.754514,-0.056350,0.408774,-0.632199,0.447418,0.146990,-0.191699,-0.581418,0.094146,-0.683263,-0.770313,0.366571,0.037516,-0.149506,0.177500,-0.413479,-0.258291,-0.283906,-0.197309,0.576581,-0.235953,-0.749292,-0.158372,-0.048512,-0.610449,-0.551893,-0.346823,0.101966,-0.096490,-0.690421,0.005322,0.537406,-0.519928,-0.930608,0.064039,0.409871,-0.148330,0.048655,-0.471792,0.483012,0.475518,-0.374640,-0.072953,-0.024701,-0.332492,-0.716757,0.494076,0.674929,-0.084504,-0.013192,-0.992656,0.669487,-0.204460,0.291221,-0.219413,-0.325105,-0.194876,-0.357657,0.376500,0.119631,-0.443717,-0.002278,0.569721,0.649409,0.461895,0.472180,-0.651565,0.081806,-0.666641,0.186706,-0.056327,-0.400799,-0.438804,0.479944,-0.368781,-0.128277,0.102031,0.737860,-0.384779,-0.112553,-0.355627,-0.610520,-0.046319,0.500558,-0.048919,-0.027259,0.485534,-0.114180,-0.014046,0.298979,-0.059359,-0.857693,-0.464566,0.677154,0.199837,0.008947,-0.418890,-0.037592,-0.097270,0.586556,0.644407,-0.066439,0.521927,0.418663,-0.379745,0.388843,0.652717,-0.119575,0.569051,-0.346798,0.440210,-0.770488,0.035754,0.435891,0.319164,0.192829,0.145809,-0.692996,-0.097430,-0.789142,-0.595334,0.143322,0.290972,-0.526300,0.008059,0.069378,-0.085284,0.173010,-0.648510,-0.082692,-0.807578,-0.210521,-0.294406,0.883326,-0.400215,0.350971,-0.375869,-0.461574,-0.559590,-0.115607,-0.736141,-0.612243,-0.257256,-0.353136,0.531404,-0.174399,-0.071538,-0.074397,-0.212377,-0.396602,-0.056769,0.288303,0.918241,-0.698133,-0.368108,-0.032372,0.293478,-0.659575,0.043859,-0.354941,0.134838,-0.777424,0.190671,-0.271562,0.067491,-0.114192,-0.437774,-0.268267,0.314446,-0.138901,-0.414110,0.127397,-0.371494,0.585448,-0.084285,0.220599,-0.492964,0.293073,0.024453,0.685054,0.687745,0.966112,-0.840094,0.889024,0.479549,-0.152352,0.626172,-0.340516,0.220612,-0.498277,0.534144,-0.763011,-0.514219,-0.341990,0.077433,-0.438915,0.403457,-0.055090,0.837996,-0.064308,
-0.799253,-0.727048,-0.843109,0.795262,-0.475906,0.393349,-0.755119,-0.861285,0.111597,-0.432260,0.758648,-0.557130,-0.439126,0.817675,-0.537525,0.944166,0.769728,-0.338408,-0.557865,-0.655666,-0.096438,1.137407,-0.270947,0.275050,-0.168962,0.825422,0.486101,0.065834,-0.490330,-0.614343,-1.024551,0.310881,0.855080,0.960913,0.547082,-0.917574,-0.844656,-0.898791,-0.756893,1.146078,-0.783454,-0.147231,-0.754787,-0.252134,0.330084,-0.480372,-0.186770,0.217775,-0.312690,0.804367,-0.594207,-0.869307,-0.550059,-0.266606,0.850472,1.188545,0.218627,-0.594475,1.019636,-0.638439,0.141460,-0.656245,0.668037,-1.143698,0.353601,-1.123911,-0.531050,0.841868,0.814344,-0.599366,-0.345945,0.160537,0.427768,-0.844894,-0.844636,1.166169,0.510018,-1.209830,-0.687702,-0.325317,-0.961013,-1.183515,-0.610744,-0.670244,0.738905,-1.095224,0.783521,0.038428,-1.024473,-0.749252,0.970441,0.167569,-0.747894,-0.399969,-1.234324,0.306741,0.177517,-1.079486,0.242549,-0.536509,-0.029195,-0.967330,1.013857,0.947617,0.764214,-0.800339,-0.972465,1.097798,0.331667,0.558144,0.243817,0.597477,-0.986698,-0.825187,0.505441,0.363249,0.249864,-0.882789,0.904228,1.200951,0.602425,0.862846,-1.237439,0.786043,-0.014854,-0.372887,0.012693,-1.170067,0.450890,0.326337,-1.022479,0.572975,0.876091,0.734416,0.522479,-0.972766,0.271506,-0.724377,-0.691904,0.900882,-0.894872,-0.773762,1.136564,-0.635546,0.615038,0.648189,0.705858,-1.147058,-1.030985,0.821337,0.569457,-0.586093,0.507973,-0.159628,-0.912561,0.638609,1.098449,-0.945705,-0.102181,0.313825,-0.930896,-0.288445,0.944879,0.900116,0.520100,-0.196517,1.197361,-0.690036,0.625824,0.955951,1.026997,0.782939,-0.667314,-1.231443,-0.662574,-0.983693,-1.257255,0.827943,0.706341,-0.949402,0.223584,-0.343414,-0.909727,1.025886,-0.974264,0.411344,-0.974105,-0.633207,-0.035608,0.930507,-0.457229,0.849560,-0.564658,-0.424837,-0.775775,-1.000156,-0.919793,-0.066490,-0.307543,-0.919598,0.103176,-0.981409,0.724745,0.383401,-0.359890,-0.778764,0.196593,0.140988,0.929144,-0.513997,-0.936399,0.508219,0.221513,-0.216667,-0.827729,0.241814,-0.575359,-0.630851,-0.329980,-0.866516,-0.312988,-0.454707,0.133245,-0.729363,0.627078,-0.469213,0.383146,-0.608092,-1.035711,0.630319,-0.567061,-0.307288,-0.916503,0.458409,-0.285447,0.705111,1.007345,1.003475,-1.158276,0.789454,0.231565,0.313858,1.135162,-0.114144,-0.747570,-1.155706,0.918183,-0.509474,-0.791859,-0.659559,0.856658,0.558477,0.979192,-0.253085,0.520058,-0.865633,
-0.613553,-0.771651,-1.060399,0.832242,-0.139686,0.123845,-0.732023,-0.826136,-0.239480,-0.459712,0.674028,-0.089136,-0.637182,0.609923,-0.633175,1.022824,0.776909,-0.266756,-0.650977,-0.691058,0.168837,0.854467,-0.223188,0.482823,-0.245277,0.963765,0.697155,0.041355,-0.194068,-0.301642,-0.689451,0.611902,1.042308,0.669821,0.684062,-0.680187,-0.748503,-0.748897,-0.924936,1.217438,-0.535381,0.240626,-0.922022,-0.473514,0.006100,-0.742850,-0.190197,0.320740,-0.659110,0.459673,-0.710790,-0.842950,-0.265272,-0.514235,0.863617,1.117145,-0.102466,-0.719778,0.825110,-0.406541,-0.227882,-0.830273,0.659627,-1.078135,-0.081481,-1.031402,-0.398138,0.767558,0.804360,-0.287964,-0.263349,0.267187,0.596634,-1.046583,-0.641683,1.176900,0.606469,-1.132482,-0.743396,-0.588360,-0.862022,-0.985345,-0.056598,-0.910519,0.879584,-0.988880,0.782075,-0.305607,-1.011849,-0.620075,1.003479,-0.172161,-0.606954,-0.040997,-1.175502,0.440737,-0.284106,-0.844093,0.530464,-0.130952,0.241744,-0.506076,0.670162,0.748446,0.364691,-0.817638,-0.775809,0.948662,0.563913,0.731296,0.420666,0.706351,-1.020180,-0.899703,0.590555,-0.059244,0.521081,-0.955541,0.858683,0.952535,0.627611,0.961497,-1.173662,0.736043,0.129826,-0.427844,0.061676,-1.003253,0.550054,-0.027371,-0.762038,0.473644,0.771904,0.388528,0.613568,-1.070194,0.457510,-0.649606,-0.940121,0.604751,-0.501067,-0.886958,0.878739,-0.854247,0.761705,0.299346,0.859259,-0.999050,-0.845798,0.470918,0.662992,-0.099394,0.636808,-0.178788,-0.655127,0.452639,1.177059,-0.782344,-0.397276,0.445320,-1.097071,-0.265768,0.674818,0.538035,0.191033,0.183590,1.103083,-0.435484,0.614382,0.695481,1.002756,0.679869,-0.620289,-1.157220,-0.513965,-0.769148,-1.016568,0.914591,0.638328,-1.016606,0.189718,-0.597909,-0.780577,0.777410,-0.686926,0.481444,-0.719657,-0.716166,0.122203,0.720304,0.044476,0.642291,-0.514791,-0.174764,-0.517622,-0.642350,-0.564895,0.001538,0.167082,-0.543288,-0.348911,-0.880624,0.428465,0.394342,-0.405169,-0.593660,0.115856,-0.113164,0.682882,-0.479856,-0.898726,0.795374,0.238256,0.098220,-0.686109,0.322854,-0.388738,-0.305088,-0.334038,-0.911420,-0.148073,-0.780832,-0.235040,-0.498793,0.861210,-0.537174,0.440459,-0.686547,-0.812404,0.498065,-0.229595,-0.115743,-0.551399,0.156217,-0.041003,0.424431,0.803661,0.872668,-0.904220,0.492310,0.251604,-0.012409,0.977803,0.130375,-0.763133,-0.920274,0.671753,-0.233036,-0.621238,-0.394503,0.773195,0.572426,1.014506,0.098487,0.072039,-0.850545,
-0.857721,-0.269886,-1.028728,0.844985,0.026968,0.681019,-0.140170,-0.134862,-0.801248,-0.608182,0.435723,-0.230244,-0.053910,0.600692,-0.146635,0.827427,0.895069,-0.714921,-0.981663,-0.735515,0.281816,0.351364,0.441560,0.759522,0.044983,0.976831,0.240905,0.868153,-0.322577,-0.302837,-0.236144,-0.047859,0.531173,0.078199,-0.108980,-0.632126,-0.210060,-0.597673,-0.552304,0.536795,-0.594558,0.056997,-0.631223,-0.174816,-0.374522,-0.910821,0.677733,0.710353,0.138812,0.567030,-0.747748,-0.976327,0.245049,-0.645967,0.692530,0.689475,-0.121867,-0.792856,0.068580,0.139298,-0.504924,-0.089803,0.227324,-0.375714,0.189807,-0.820788,-0.183195,0.311723,0.529465,0.094408,-0.434500,0.333874,0.388159,-0.606897,-0.129758,0.588189,0.037298,-0.459584,0.043569,-0.817456,-0.667213,-0.564225,-0.092106,-0.706116,0.645047,-0.677851,0.842101,-0.439103,-0.231220,-0.676757,0.519748,-0.278860,0.181058,-0.225610,-0.847785,0.800069,-0.098286,-0.776197,0.769545,0.196586,0.074448,-0.375401,0.721744,0.014347,-0.347258,-0.374021,-0.503255,0.519806,-0.064122,1.027508,0.413439,0.519769,-0.771521,-1.140062,0.816153,0.454008,0.306522,-0.813450,0.893423,0.580976,1.013199,0.960040,-0.913272,0.289651,-0.668056,0.031427,-0.229590,-0.678558,0.361944,-0.388424,-0.326622,-0.387210,0.229219,0.516476,0.411892,-0.652163,0.258958,0.251499,-0.506711,-0.164450,-0.367080,-0.782067,0.140396,-0.774306,0.743906,-0.304037,0.478403,-0.711861,-0.335077,0.575521,0.533326,-0.129826,0.527449,-0.333872,-0.171884,0.142678,0.872137,-0.264474,0.083499,1.049848,-0.443663,-0.632467,-0.047328,0.203242,0.175940,-0.407390,0.772624,-0.230463,0.630147,0.234651,0.654114,0.536944,0.004690,-0.706064,0.503304,-0.579160,-0.790923,0.988169,0.775191,-0.839426,0.385596,-0.361154,-0.601795,0.152288,-0.302007,-0.130500,-0.123810,-0.926973,-0.473590,0.431855,0.176793,-0.143573,-0.981843,0.464041,-0.807054,-0.078165,-0.575052,0.143620,-0.059266,0.042682,0.193618,-0.224254,0.161864,0.726800,-0.556850,-0.110281,0.605290,0.502609,0.321186,-0.926886,-0.679788,0.456161,0.682655,0.033024,-0.398393,0.179639,-0.002077,-0.685387,0.532110,-0.516292,0.083356,-0.449460,-0.186518,-0.211138,0.250710,-0.970424,0.067557,-0.754952,-0.444155,0.529691,-0.381596,0.506561,-0.520296,-0.616188,0.031687,0.192007,0.455220,0.643533,-0.853565,0.779634,0.819637,-0.813782,0.419094,0.337926,-0.475357,-0.271111,0.104996,-0.598201,0.193894,-0.919340,-0.024160,0.136609,0.485824,-0.441972,0.246838,-0.961278,
-0.355262,-0.674172,-1.006665,0.921652,-0.173300,0.147207,-0.549367,-0.880585,-0.350712,-0.611093,0.437150,0.065159,-0.445853,0.683153,-0.764102,1.079506,0.496151,-0.040681,-0.564944,-0.315139,0.305682,0.605600,-0.548623,0.451460,-0.293169,1.110823,0.667592,-0.013494,0.288794,-0.470873,-0.704134,0.451416,0.944364,0.309637,0.502540,-0.554887,-0.662965,-0.614207,-0.720958,1.101199,-0.729287,0.546414,-0.601757,-0.594585,-0.025965,-0.847889,-0.068003,0.030163,-0.498509,0.071549,-0.622695,-0.602754,-0.182175,-0.601467,0.845101,0.914953,-0.052969,-0.515212,0.724028,-0.315625,-0.477774,-0.778638,0.689341,-0.852102,-0.084573,-1.153709,-0.456756,0.913550,0.540621,-0.189428,-0.156745,-0.085072,0.620849,-1.015307,-0.837134,0.888783,0.534944,-0.864873,-0.586735,-0.449867,-0.662635,-0.936403,0.287969,-0.786631,0.595738,-0.771255,0.487748,-0.554127,-0.814241,-0.615863,0.919885,0.179711,-0.207636,0.242818,-1.018784,0.054202,-0.427296,-1.023399,0.721737,-0.111684,0.163563,-0.638747,0.468352,0.669294,0.111584,-0.848422,-0.802601,0.777322,0.740041,0.742507,0.525043,0.638756,-1.065014,-0.711649,0.644716,-0.546741,0.566122,-0.938335,0.825946,0.816860,0.443893,0.854447,-0.940730,0.681823,-0.080464,-0.630756,0.532251,-0.900845,0.690450,-0.143415,-0.601142,0.330399,0.693178,0.167603,0.396436,-0.971738,0.594544,-0.506900,-0.826339,0.574398,-0.193067,-0.673421,0.650777,-0.707622,0.634517,0.159700,0.937445,-0.734607,-0.871920,0.096351,0.754378,0.268832,0.662394,-0.439640,-0.683285,-0.006791,0.895424,-0.590108,-0.437236,0.730259,-1.139917,-0.243273,0.287522,0.244784,-0.224529,0.074496,0.932730,-0.353737,0.848743,0.738708,0.790824,0.592153,-0.508520,-1.078551,-0.306638,-0.447537,-0.823923,0.781228,0.556509,-0.888945,-0.156104,-0.447475,-0.731564,0.491417,-0.438872,0.095257,-0.600179,-0.782504,0.611905,0.696362,0.010639,0.603170,-0.557546,-0.204956,-0.196414,-0.608286,-0.228738,0.070964,0.175245,-0.345506,-0.271406,-0.637172,0.371608,0.337254,0.079973,-0.478613,0.240220,-0.471863,0.516124,-0.085525,-0.871169,0.922145,0.013830,0.245812,-0.654681,0.593775,-0.321996,0.049818,-0.339760,-0.611860,0.056866,-0.737808,-0.281247,-0.764844,0.733460,-0.288724,0.313062,-0.454944,-0.529548,0.030067,-0.184593,-0.397091,-0.065719,0.214114,-0.388431,0.365663,0.495087,0.564759,-0.643989,0.127631,0.531119,-0.017885,0.608989,0.430052,-0.782878,-0.689926,0.487420,-0.169270,-0.718314,-0.328901,0.449517,0.411080,0.640786,0.326963,0.016097,-0.757920,
-0.547355,-0.531183,-0.926972,0.848936,0.098874,-0.155175,-0.580676,-0.819936,-0.121972,-0.620679,0.322370,0.164037,-0.395843,0.494017,-0.559104,1.085195,0.421667,-0.097191,-0.493428,-0.573994,0.374693,0.476619,-0.369808,0.523160,-0.124716,1.085038,0.626077,0.053607,0.262625,-0.476778,-0.663808,0.556589,0.917632,0.212328,0.436045,-0.334251,-0.644304,-0.538831,-0.889031,0.946108,-0.716148,0.701914,-0.766725,-0.610074,-0.221676,-0.917034,0.052510,-0.028146,-0.508943,0.081987,-0.606739,-0.519992,-0.323760,-0.765433,0.665932,0.866560,-0.352359,-0.671171,0.616109,-0.364218,-0.520783,-0.754023,0.710802,-0.822664,-0.239444,-1.146101,-0.363675,0.889719,0.733826,-0.275823,-0.149291,0.121293,0.771369,-1.072895,-0.706176,0.825719,0.692352,-0.749913,-0.649218,-0.694814,-0.406564,-0.835274,0.293189,-0.862313,0.643099,-0.732249,0.623167,-0.629849,-0.819246,-0.335623,0.810830,-0.090100,-0.137805,0.016571,-1.011085,-0.006090,-0.603266,-0.870279,0.742162,-0.241146,0.163900,-0.475993,0.464489,0.744219,0.258629,-0.791652,-0.611906,0.655840,0.685879,0.735983,0.635992,0.720131,-1.096945,-0.861407,0.465999,-0.438492,0.647497,-1.060269,0.795448,0.855832,0.450533,0.711677,-0.879525,0.758099,-0.006093,-0.399936,0.537664,-0.946129,0.720872,-0.223892,-0.437955,0.424266,0.796973,0.003072,0.407505,-0.971074,0.675860,-0.353302,-0.725340,0.358107,-0.301034,-0.679433,0.705011,-0.593928,0.778899,0.069080,1.008946,-0.586328,-0.871478,0.077398,0.669754,0.116638,0.781111,-0.475627,-0.560535,-0.072074,0.818297,-0.401689,-0.575092,0.673187,-1.059362,-0.237643,0.204152,0.161659,-0.413234,0.311775,0.852545,-0.186121,0.783430,0.722447,0.758364,0.707848,-0.644434,-1.005749,-0.364772,-0.383674,-0.751668,0.821400,0.686953,-0.750542,-0.198278,-0.581255,-0.776860,0.301808,-0.382536,0.173496,-0.422435,-0.875176,0.523412,0.441152,-0.101940,0.610886,-0.703550,0.020333,-0.281098,-0.580713,-0.027990,0.250393,0.067837,-0.280022,-0.525895,-0.481754,0.431030,0.355604,0.023028,-0.235899,0.295127,-0.374168,0.362531,-0.099167,-0.694442,0.964482,-0.106648,0.477433,-0.736715,0.626808,-0.448903,0.030427,-0.388420,-0.660056,0.185531,-0.707773,-0.014073,-0.750370,0.594118,-0.383471,0.457938,-0.473932,-0.387038,-0.093921,-0.011072,-0.429821,0.026034,0.024350,-0.414970,0.191020,0.310214,0.505267,-0.496453,0.032019,0.384025,-0.031852,0.585099,0.432206,-0.829695,-0.675585,0.418670,-0.031210,-0.644870,-0.224793,0.395321,0.277508,0.699109,0.170287,-0.219935,-0.882188,
-0.744598,-0.300032,-0.841682,0.298316,-0.756722,0.636755,-0.750876,-0.999696,-0.160407,-0.000161,1.112123,-0.581469,-0.022887,1.064522,0.001653,0.574410,0.542118,-0.532467,-0.670737,-0.156551,-0.226867,0.771735,-0.185747,-0.078616,-0.819916,0.573004,-0.021122,-0.176894,-0.271186,-0.595476,-0.651281,0.070262,0.349450,0.564022,-0.107711,-1.127898,-0.179386,-0.984179,-0.208293,0.949742,-0.834474,-0.300635,-0.276377,0.239024,0.742469,-0.428003,-0.003393,0.451538,0.056681,0.934510,0.077334,-0.784051,0.038366,0.182166,0.391165,0.755973,0.681360,-0.480092,0.584729,-0.929430,0.390918,-0.341882,0.075988,-1.073985,0.138648,-0.888676,-1.039734,0.630671,0.391090,-0.330397,0.446981,-0.505530,0.287809,-0.781564,-0.507517,0.888877,0.227633,-0.966254,-0.558755,-0.153271,-0.617096,-0.954273,-0.501261,-0.439512,0.315891,-1.046010,0.271222,0.040053,-0.849914,-0.990853,0.409674,0.594444,-0.692908,0.304329,-0.826731,0.615615,0.552648,-0.689158,0.210202,-0.426303,-0.584889,-0.868263,0.779200,0.942184,0.172455,-0.616786,-1.153750,1.132855,-0.069841,0.015292,-0.253542,0.035932,-0.522496,-0.542855,0.346398,0.210150,-0.008239,-0.599748,0.858209,1.047475,0.666099,0.624754,-0.890680,0.632915,-0.567541,0.210993,0.218117,-0.761710,-0.246460,0.771597,-0.529777,0.147550,0.601102,0.984488,-0.221081,-0.565982,0.204779,-0.819576,-0.515226,0.831175,-0.535816,-0.514854,0.742662,-0.339860,-0.006383,0.759575,0.260030,-1.175294,-0.937293,0.861320,-0.039936,-0.086423,-0.203751,0.369714,-0.422921,0.763174,0.758653,-0.357669,0.315011,0.510641,-0.600394,0.200673,0.847299,0.270216,0.231903,-0.330279,0.851683,-1.090326,0.073986,0.487931,0.805421,0.520007,-0.078825,-0.815010,-0.391052,-0.896168,-1.056522,0.271278,0.271076,-1.030157,-0.322810,-0.423396,-0.566868,0.624831,-0.872437,-0.087248,-0.995453,-0.516774,-0.278373,0.981112,-0.546141,0.586442,-0.232325,-0.269475,-0.774891,-0.482103,-0.821027,-0.648474,-0.410527,-0.643889,0.355336,-0.627907,0.495124,0.260306,-0.241420,-0.840346,0.515332,0.359459,0.788442,-0.553358,-0.885212,0.154742,0.054954,-0.625626,-0.379774,-0.333810,-0.282920,-0.763649,0.059562,-0.367288,-0.536603,-0.293016,-0.409565,-0.575153,0.594367,-0.130650,-0.316063,-0.267818,-0.841616,0.254596,-0.317202,-0.125887,-0.641567,0.277895,-0.301805,1.027714,0.647351,1.173518,-1.099582,0.842671,0.577674,0.188125,1.098093,0.084446,-0.171626,-0.812504,0.935503,-0.920276,-0.613325,-0.491250,0.452903,0.130843,0.759294,-0.382900,0.702572,-0.459417,
-0.528658,-0.599600,-0.854581,0.763022,-0.152151,0.158643,-0.707745,-0.699275,-0.046574,-0.526929,0.522463,-0.180972,-0.424847,0.333152,-0.592601,1.034619,0.631338,-0.435438,-0.696728,-0.680563,0.125190,0.699926,-0.052599,0.560921,0.031063,0.954080,0.767378,0.210991,-0.288933,-0.130933,-0.446888,0.412672,0.977874,0.506101,0.626378,-0.484465,-0.587238,-0.718745,-0.795304,1.130460,-0.260066,0.245060,-0.823873,-0.440396,-0.138193,-0.743074,-0.211192,0.268020,-0.703510,0.268215,-0.679024,-0.858503,-0.180787,-0.549086,0.675960,0.976751,-0.196858,-0.602358,0.788700,-0.141065,-0.121438,-0.612470,0.570416,-0.931193,-0.268385,-0.879511,-0.114434,0.730301,0.675179,-0.054802,-0.454369,0.347439,0.656181,-0.839155,-0.440691,0.985240,0.506038,-1.003018,-0.648034,-0.502587,-0.772023,-0.779817,-0.138836,-0.906237,0.906883,-0.777982,0.836582,-0.364881,-0.836630,-0.544898,0.927914,-0.266499,-0.472915,-0.107475,-1.090980,0.266537,-0.319082,-0.797205,0.617294,0.068106,0.463198,-0.585078,0.472906,0.618519,0.365522,-0.626580,-0.550863,0.797444,0.532817,0.726510,0.340494,0.627632,-0.963506,-0.904647,0.707090,-0.066204,0.496794,-0.970339,0.611562,0.824862,0.665259,0.958796,-1.160798,0.782085,0.006910,-0.520953,0.094957,-0.812870,0.586358,-0.170732,-0.773041,0.321783,0.606372,0.245693,0.698512,-0.996710,0.540109,-0.455684,-0.778555,0.498405,-0.466527,-0.730696,0.781678,-0.907989,0.831505,0.340765,0.917214,-0.885416,-0.604710,0.330373,0.652485,-0.212007,0.676035,-0.104370,-0.591740,0.353240,1.031360,-0.623054,-0.458280,0.547737,-1.025820,-0.391233,0.531156,0.487708,0.091652,0.445520,1.041558,-0.282017,0.594345,0.452508,0.791655,0.541954,-0.463629,-0.978959,-0.330014,-0.716305,-0.947874,0.973870,0.517958,-0.768581,0.259376,-0.632638,-0.652482,0.630943,-0.573360,0.546186,-0.651567,-0.660659,0.124538,0.549971,-0.040150,0.642621,-0.471686,-0.286414,-0.317184,-0.479816,-0.465677,0.128794,0.092279,-0.541746,-0.345378,-0.652719,0.234397,0.419271,-0.258344,-0.469483,0.212502,-0.036413,0.579658,-0.435458,-0.717849,0.839818,0.355480,0.195737,-0.633040,0.394672,-0.360336,-0.072486,-0.426847,-0.875134,-0.004197,-0.692327,-0.119034,-0.411441,0.749993,-0.590400,0.367901,-0.714235,-0.640228,0.339103,0.033836,-0.079276,-0.438883,0.272596,0.070506,0.254110,0.654813,0.685894,-0.780819,0.380485,0.416614,-0.020318,0.838644,-0.029565,-0.782626,-0.777248,0.485999,-0.186700,-0.530256,-0.373251,0.441813,0.496900,0.846091,0.205015,-0.057818,-0.896007,
-0.840835,-0.535993,-0.666907,0.575605,-0.034289,-0.092792,-0.363377,-0.826269,0.121404,-0.702409,0.471406,-0.269397,-0.480538,0.670530,-0.659803,1.103270,0.387733,0.142123,-0.265114,-0.716569,0.214280,0.637209,-0.176892,0.261119,0.085650,0.708265,0.468948,-0.019876,0.019009,-0.461688,-0.797562,0.409699,0.724392,0.426591,0.379718,-0.554648,-0.873154,-0.637301,-0.687720,0.920004,-0.669697,0.294796,-0.630119,-0.454374,-0.070706,-0.684982,0.057995,-0.086061,-0.305875,0.395002,-0.518224,-0.355704,-0.599614,-0.582809,0.831732,0.894690,-0.186520,-0.833030,0.760108,-0.385969,-0.116139,-0.725702,0.660294,-0.964962,0.299874,-0.916834,-0.318595,0.967085,0.680856,-0.560278,-0.178815,0.193942,0.697007,-0.885206,-0.651489,0.843685,0.695254,-0.869075,-0.690780,-0.602602,-0.407545,-0.944088,-0.130968,-0.750900,0.631142,-0.877220,0.601842,-0.406589,-0.802066,-0.251907,0.789610,0.012857,-0.321366,-0.210087,-1.014402,0.211976,-0.189957,-0.907503,0.571766,-0.449649,-0.072211,-0.589425,0.753664,0.801965,0.462587,-0.760429,-0.698894,0.850859,0.339965,0.564429,0.617685,0.675525,-0.920639,-0.765738,0.331553,-0.062940,0.470721,-1.059228,0.782044,0.877862,0.138983,0.516435,-0.929852,0.664410,0.092578,-0.147422,0.423153,-1.046916,0.603540,0.015290,-0.615189,0.608593,0.926827,0.336971,0.507042,-1.030236,0.480855,-0.465546,-0.554471,0.416793,-0.668548,-0.522634,0.958787,-0.395348,0.704042,0.183671,0.917835,-0.801945,-0.936606,0.505118,0.361407,-0.196288,0.622607,-0.139955,-0.853723,0.201038,0.781749,-0.539770,-0.515030,0.350712,-0.908889,-0.084308,0.439855,0.482093,0.002425,0.010776,1.007166,-0.381032,0.763397,0.859439,0.796554,0.828104,-0.629072,-0.948099,-0.533375,-0.549460,-0.977951,0.729909,0.627924,-0.800073,0.074890,-0.468668,-0.949361,0.551863,-0.730100,0.105678,-0.586714,-0.496146,0.261497,0.380274,-0.220128,0.746979,-0.335144,-0.126400,-0.504608,-0.744028,-0.348790,0.332877,0.234161,-0.582656,-0.194401,-0.607399,0.598424,0.562737,0.025559,-0.352016,0.190653,-0.272412,0.379195,-0.227368,-0.722717,0.726221,-0.141742,0.155910,-0.829017,0.307676,-0.579607,-0.291035,-0.142071,-0.582725,-0.227785,-0.807624,0.321381,-0.660759,0.290191,-0.232168,0.593893,-0.366151,-0.593274,-0.055545,-0.481582,-0.445728,-0.448106,0.236404,-0.348222,0.413658,0.492064,0.591749,-0.774276,0.374811,0.183168,0.120055,0.772290,0.107573,-0.851249,-0.741048,0.623493,-0.152981,-0.683845,-0.511575,0.464500,0.415924,0.481227,-0.114858,-0.101164,-0.723291
  };
  float b[10*256];
  cv::Mat src1(10, 256, CV_32F, a);
  cv::Mat src2(10, 256, CV_32F, b);
  cv::Mat dst;

  FILE *fp;
  fopen_s(&fp, "src2.txt", "rt");
  float f;
  for (int i = 0; i < 10*256; ++i)
  {
    fscanf_s(fp, "%f", &f);
    b[i] = f;
  }
  fclose(fp);

  //printf("src1 ----------------------------\n");
  //printMat(src1);
  //printf("src2 ----------------------------\n");
  //printMat(src2);

  for (int i = 0; i < 10; ++i)
    matMul(src1, src2, 1, dst, cv::GEMM_1_T);

  //printf("dst -----------------------------\n");
  //printMat(dst);

  return 0;
#endif

	InitializeLog();

#if 0 // s-jis 漢字一覧テキストファイルの作成
	MakeSjisKanjiList(40);
	return 0;
#elif 0 // アフィン変換により新たなサンプル画像を生成する。
	CreateTransformedSamples("trainData-kanaKigou");
	return 0;
#elif 0 // テスト：文字の相似度を計算する。
	cv::Mat templImage, sampleImage;
	
	templImage = cv::imread("template\\ふ1.png", cv::IMREAD_GRAYSCALE);
	GetNormalizedImage(sampleImage, "testData\\ぶ-IMG_8154-00200.png", 64, 64, 150);
	
	float f = MatchingResidue(templImage, sampleImage);
	printf("残差：%f\n", f);
	return 0;
#elif 0 // テンプレート文字画像の正規化をおこなう。
	NormalizeTemplates("template");
	return 0;
#endif

  for (int i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], "-v") == 0)
      param.verbose = true;
    else if (strcmp(argv[i], "-f") == 0)
      param.firstLayerToTrain = atoi(argv[++i]);
    else if (strcmp(argv[i], "-l") == 0)
      param.lastLayerToTrain = atoi(argv[++i]);
    else if (strcmp(argv[i], "-h") == 0)
      param.imageHeight = atoi(argv[++i]);
    else if (strcmp(argv[i], "-w") == 0)
      param.imageWidth = atoi(argv[++i]);
    else if (strcmp(argv[i], "-i") == 0)
      param.maxIter = atoi(argv[++i]);
    else if (strcmp(argv[i], "-a") == 0)
      param.randomAffineTransform = true;
    else if (strcmp(argv[i], "-b") == 0)
      param.blackBackground = true;
    else if (strcmp(argv[i], "-c") == 0)
      param.cutOutBlackRegion = true;
    else if (strcmp(argv[i], "-C") == 0)
      param.cumulative = true;
    else if (strcmp(argv[i], "-e") == 0)
      param.supportedExtensionList = argv[++i];
    else if (strcmp(argv[i], "-L") == 0)
      param.layerParamStr = argv[++i];
    else if (strcmp(argv[i], "-p") == 0)
      param.paramFileName = argv[++i];
    else if (strcmp(argv[i], "-E") == 0)
      param.evaluateEvery = atoi(argv[++i]);
    else if (strcmp(argv[i], "-m") == 0)
    {
      std::istringstream seg(argv[++i]);
      std::string token;
  	  getline(seg, token, ',');

      if (token == "BPROP")
      {
        param.update_param.type = updateParam::bprop;
        while (true) // never loops
        {
          if (!getline(seg, token, ','))
            break;
          param.update_param.learningRate = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.finalLearningRate = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.learningRateDecay = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.initMomentum = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.finalMomentum = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.momentumDecayEpoch = atoi(token.c_str());
        }
      }
      else if (token == "RPROP")
      {
        param.update_param.type = updateParam::rprop;
        while (true) // never loops
        {
          if (!getline(seg, token, ','))
            break;
          param.update_param.dw0 = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.dw_plus = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.dw_minus = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.dw_max = (real)atof(token.c_str());
          if (!getline(seg, token, ','))
            break;
          param.update_param.dw_min = (real)atof(token.c_str());
        }
      }
      else
      {
        Log("Error: unknown training method: %s\n", token.c_str());
        param.command.clear();
        break;
      }
    }
    else if (argv[i][0] == '-')
    {
      Log("Error: unknown option: %s\n", argv[i]);
      break;
    }
    else if(param.command.empty())
      param.command = argv[i];
    else
    {
      if(param.dataDir.empty())
      {
        param.dataDir = argv[i];
        if (param.dataDir[param.dataDir.size() - 1] != '\\')
          param.dataDir += '\\';
      }
      else
      {
        param.dataDir2 = argv[i];
        if (param.dataDir2[param.dataDir2.size() - 1] != '\\')
          param.dataDir2 += '\\';
      }
    }
  }

  if (param.command.empty() || (param.command != "WIMAGE" && param.dataDir.empty()))
  {
    PrintUsage(argv[0]);
  	FinalizeLog();
    return 1;
  }

	SYSTEMTIME sysTime;
	GetLocalTime( &sysTime );
	Log("\n%d/%02d/%02d %02d:%02d:%02d %s started.\n",
		sysTime.wYear, sysTime.wMonth, sysTime.wDay,
		sysTime.wHour, sysTime.wMinute, sysTime.wSecond,
    argv[0]);
  Log("real type is %s.\n", realStr);
  param.log();

  NeuralNet nn;

  int ret = 0;

  if (param.command == "TRAIN" || param.command == "AE")
  {
    while (true) // never loops
    {
      // weight file があれば読み込む。なければ param.layerParamStr から構築する。
      if (BuildNN(nn, param.paramFileName.c_str()) != 0)
        break;

      unsigned _int16 firstCharCode;
      int class_count;
      const int numData = CountFilesInDir(param.dataDir, param.supportedExtensionList, firstCharCode, class_count);
      Log("Number of data is %d.\n", numData);
      Log("  First character   : '%s'.\n", charStr(firstCharCode));
      Log("  Number of classes : %d.\n", class_count);

      if (nn.firstCharCode == 0) // built from layer param string
      {
        nn.firstCharCode = firstCharCode;
      }
      else // built from weight file
      {
        if (firstCharCode < nn.firstCharCode)
        {
    	    Log("Error: the first char code in data < nn.firstCharCode.\n");
          ret = 2;
          break;
        }
        firstCharCode = nn.firstCharCode;
      }

      if (nn.outSize() < class_count)
      {
    	  Log("Error: number of classes in data exceeds outSize of NN.\n");
        ret = 3;
        break;
      }
      class_count = nn.outSize();

      cv::Mat trainData, responses, sampleWeights;
      int retcode = ReadTrainData(
        trainData,
        responses,
        sampleWeights,
        param.dataDir,
        param.supportedExtensionList,
        numData,
        firstCharCode,
        class_count);
      if (retcode != 0)
      {
    	  Log("Error: ReadTrainData() returned%d.\n", retcode);
        ret = 4;
        break;
      }

      if (param.command == "TRAIN")
      {
	      // 1. unroll the responses
	      Log("Unrolling the responses...\n");

        // set new_responses (1-of-K cording)
	      cv::Mat new_responses(numData, class_count, CV_REAL);
        new_responses = 0;
	      for (int i = 0; i < numData; ++i)
	      {
		      int cls_label = (unsigned _int32)responses.at<_int32>(i, 0) - firstCharCode;
		      if (cls_label < class_count)
  		      new_responses.at<real>(i, cls_label) = 1;
        }

	      // 2. train classifier
        Log("Training by %s...\n", (param.update_param.type == updateParam::bprop)? "BPROP" : "RPROP");
        double t2, t1 = (double)cv::getTickCount();
        real E;
        int retcode = nn.train(
          trainData,
          param.randomAffineTransform? RandomAffineTransform : NULL,
          (void *)(&param.imageHeight),
          new_responses,
          sampleWeights,
          param.firstLayerToTrain,
          param.update_param,
          param.maxIter,
          param.evaluateEvery,
          CalculateRecognitionError,
          E);
	      
        t2 = (double)cv::getTickCount();
	      Log("Time for training : %0.3lf\n", (t2 - t1) / cv::getTickFrequency());
	      t1 = t2;

        if (retcode != 0)
        {
          Log("nn.train() returned %d\n", retcode);
          ret = 5;
          break;
        }

        Log("training error = %f\n", E);
      }
      else if (param.command == "AE")
      {
        // autoencoding (unsupervised training)

        if (param.lastLayerToTrain == -1)
          param.lastLayerToTrain = nn.numLayers() - 1;

	      Log("Training by autoencoding...\n");
        double t2, t1 = (double)cv::getTickCount();
        std::vector<real> E;
        int retcode = nn.autoencode(trainData, sampleWeights, param.lastLayerToTrain, param.update_param, param.maxIter, E);

	      t2 = (double)cv::getTickCount();
	      Log("Time for training : %0.3lf\n", (t2 - t1) / cv::getTickFrequency());
	      t1 = t2;

        if (retcode != 0)
        {
          Log("nn.autoencode() returned %d\n", retcode);
          ret = 6;
          break;
        }
      }

  	  nn.writeBinary(param.paramFileName.c_str());
  	  Log("Wrote NN parameters to file : %s\n", param.paramFileName.c_str());

      break;
    }
  }
  else if (param.command == "TEST")
  {
    if (BuildNN(nn, param.paramFileName.c_str(), true) == 0) // read from weight file 
    {
  	  Log("Testing...\n");
      Predict(param.dataDir.c_str(), param.maxTestImage, nn);
    }
  }
  else if (param.command == "WIMAGE")
  {
    if (BuildNN(nn, param.paramFileName.c_str(), true) == 0) // read from weight file 
    {
      // 重みを画像に書き出す。
  	  Log("Writing wights to image...\n");
      char filename[256];
      for (int i = 0; i < nn.numLayers(); ++i)
      {
        sprintf_s(filename, "wImage%d.png", i);
        int ret2 = nn.writeWeightImage(i, filename, 0, param.cumulative);
        if (ret2 != 0)
        {
      	  Log("Cannot write wight image for layer %d. ret = %d\n", i, ret2);
          ret = 7;
        }
      }
    }
  }

  Log("Program terminated.\n");

	FinalizeLog();
	
	return ret;
}

