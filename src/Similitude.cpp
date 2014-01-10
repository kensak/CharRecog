/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "log.h"

//#define DIFF_IS_SUM_SQUARE

//#define PRINT_DEBUG_INFO

// adjustedImage の点の明るさのヒストグラムが constImage に近くなるよう明るさを LUT で変換する。
// val=0 から始めて、明るさが val 以下の点の数が constImage のそれに近づくよう、
// adjustedImage のまだ新たな明るさを割り当てていない点の暗い方から
// 新たな明るさを割り当てていく。
void MatchHistogram(const cv::Mat &constImage, cv::Mat &adjustedImage)
{
	cv::GaussianBlur(adjustedImage, adjustedImage, cv::Size(3, 3), 0.0);

	// constImage のヒストグラムを計算する。
	int histSize[] = {256};
	float ranges_[] = { 0, 256 };
	const float* ranges[] = { ranges_ };
	cv::Mat templHist;
	int channels[] = {0};
	calcHist(&constImage, 1, channels, cv::Mat(), templHist, 1, histSize, ranges, true, false);
	
	// その累積を計算する。
	cv::Mat accumulatedTemplHist(1, 256, CV_16U);
	unsigned _int16 sum = 0;
	for (int i = 0; i < templHist.rows; ++i)
	{
		sum += (unsigned _int16)templHist.at<float>(i);
		accumulatedTemplHist.at<unsigned _int16>(0, i) = sum;
	}
	// std::cout << accumulatedTemplHist << std::endl;

	// adjustedImage のヒストグラムを計算する。
	cv::Mat sampleHist;
	calcHist(&adjustedImage, 1, channels, cv::Mat(), sampleHist, 1, histSize, ranges, true, false);

	// その累積を計算する。
	cv::Mat accumulatedSampleHist(1, 256, CV_16U);
	sum = 0;
	for (int i = 0; i < templHist.rows; ++i)
	{
		sum += (unsigned _int16)sampleHist.at<float>(i);
		accumulatedSampleHist.at<unsigned _int16>(0, i) = sum;
	}
	// std::cout << accumulatedSampleHist << std::endl;

	cv::Mat LUT(1, 256, CV_8U);
	int tIndex = 0, sIndex = 0;
	for ( ; tIndex < 256 && sIndex < 256; ++tIndex)
	{
		while(accumulatedSampleHist.at<unsigned _int16>(0, sIndex) < accumulatedTemplHist.at<unsigned _int16>(0, tIndex))
		{
			LUT.at<unsigned char>(0, sIndex) = tIndex;
			++sIndex;
		}
	}
	while (sIndex < 256)
	{
		LUT.at<unsigned char>(0, sIndex) = 255;
		++sIndex;
	}
	//std::cout << LUT << std::endl;

	adjustedImage.convertTo(adjustedImage, CV_8U);
	cv::LUT(adjustedImage, LUT, adjustedImage);
	adjustedImage.convertTo(adjustedImage, CV_32F);
}

float CalcSumSquare(const cv::Mat &constImage, const cv::Mat &adjustedImage_)
{
	cv::Mat adjustedImage = adjustedImage_.clone();

	MatchHistogram(constImage, adjustedImage);

	cv::Mat diff = adjustedImage - constImage;

#ifdef DIFF_IS_SUM_SQUARE
  // sum square
	return float(diff.dot(diff));
#else
  diff = abs(diff);
	cv::Mat A = cv::Mat::ones(diff.rows, diff.cols, diff.type());
	return static_cast<float>(diff.dot(A));
#endif
}

typedef cv::Mat_<double> MakeTransformMatrixType(int i, const cv::Mat &image);

inline cv::Mat_<double> OptimizeVertPos(int i, const cv::Mat &image)
{
	return (cv::Mat_<double>(2,3) << 1, 0, 0, 0, 1, i);
}

inline cv::Mat_<double> OptimizeHorizPos(int i, const cv::Mat &image)
{
	return (cv::Mat_<double>(2,3) << 1, 0, i, 0, 1, 0);
}

inline cv::Mat_<double> OptimizeHeight(int i, const cv::Mat &image)
{
	const double d = i * 0.5 / image.cols;
	return (cv::Mat_<double>(2,3) << 1, 0, 0, 0, 1+2*d, -d*image.rows);
}

inline cv::Mat_<double> OptimizeWidth(int i, const cv::Mat &image)
{
	const double d = i * 0.5 / image.cols;
	return (cv::Mat_<double>(2,3) << 1+2*d, 0, -d*image.cols, 0, 1, 0);
}

inline cv::Mat_<double> OptimizeRotation(int i, const cv::Mat &image)
{
	const double theta = i * 5 * 2 * M_PI / 360;
	const double cosT = cos(theta);
	const double sinT = sin(theta);
	const double cx = 0.5 * image.cols;
	const double cy = 0.5 * image.rows;
	return (cv::Mat_<double>(2,3) << cosT, -sinT, cx * (1 - cosT + sinT), sinT, cosT, cy * (1 - sinT - cosT));
}

static void Optimize(
	const cv::Mat &constImage,
	cv::Mat &adjustedImage,
	float &sumSquare,
	int &limit,
	MakeTransformMatrixType &MakeTransformMatrix
	)
{
	cv::Mat currentBestImage;
	cv::Mat transformedImage(adjustedImage.rows, adjustedImage.cols, adjustedImage.type());

	int i, sign = 1;
	for (i = 1; i <= limit; ++i)
	{
		cv::Mat M = MakeTransformMatrix(sign * i, adjustedImage);

		cv::warpAffine(adjustedImage, transformedImage, M, adjustedImage.size(),
			cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
		
		float newSumSquare = CalcSumSquare(constImage, transformedImage);
		if (newSumSquare < sumSquare)
		{
			sumSquare = newSumSquare;
			currentBestImage = transformedImage.clone(); 
		}
		else
		{
			if (sign == 1)
			{
				if (i > 1)
					break;
				else
				{
					// sign = -1 に進む。
					sign = -1;
					i = 0;
					continue;
				}
			}
			else
				break;
		}
	}

	--i;

#ifdef PRINT_DEBUG_INFO
	printf("%2d ", i);
#endif

	if (i > 0)
	{
		adjustedImage = currentBestImage;
		limit -= i;
	}
}

void WriteDiffImage(const cv::Mat &templImage, const cv::Mat &sampleImage)
{
	cv::Mat diff = templImage - sampleImage;

	std::vector<cv::Mat> channel(3);
	channel[0].create(diff.size(), CV_8U);
	channel[0] = 0;
	cv::Mat red = max(0, diff);
	red.convertTo(channel[1], CV_8U);
	cv::Mat blue = max(0, -diff);
	blue.convertTo(channel[2], CV_8U);
	cv::Mat colorImage;
	cv::merge(channel, colorImage);

	static int count = 0;
	char fileName[32];
	sprintf_s(fileName, sizeof(fileName), "work\\color%03d.bmp", count++);
	cv::imwrite(fileName, colorImage);
}

float MatchingResidue(const cv::Mat &templImage_, const cv::Mat &sampleImage_)
{
	const char *funcName = "MatchingResidue";

    CV_Assert(templImage_.type() == CV_8UC1 && sampleImage_.type() == CV_8UC1);
    CV_Assert(templImage_.cols == 64 && templImage_.rows == 64 && sampleImage_.cols == 64 && sampleImage_.rows == 64);

	// templImage と sampleImage に黒い枠をつけて大きさを 100x100 にする。
	cv::Mat templImage;
	copyMakeBorder(templImage_, templImage, 18, 18, 18, 18, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat sampleImage;
	copyMakeBorder(sampleImage_, sampleImage, 18, 18, 18, 18, cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // ピクセル値を float に変換する。
	templImage.convertTo(templImage, CV_32F);
	sampleImage.convertTo(sampleImage, CV_32F);

 	float sumSquare = CalcSumSquare(sampleImage, templImage);

	int limit = templImage.rows * 2 / 3; // templImage.rows / 2 では足りない。
	int lastLimit = limit + 1;
	while (0 < limit && limit < lastLimit)
	{
		lastLimit = limit;

		// templImage を変形して sampleImage に最もマッチするものを探す。

		// 横方向に伸縮
		Optimize(sampleImage, templImage, sumSquare, limit, OptimizeWidth);

		// 縦方向に伸縮
		Optimize(sampleImage, templImage, sumSquare, limit, OptimizeHeight);

		// 横方向に移動
		Optimize(sampleImage, templImage, sumSquare, limit, OptimizeHorizPos);

		// 縦方向に移動
		Optimize(sampleImage, templImage, sumSquare, limit, OptimizeVertPos);

		// 回転
		Optimize(sampleImage, templImage, sumSquare, limit, OptimizeRotation);

#ifdef PRINT_DEBUG_INFO
		printf("\n");
#endif
	}

#if 1
	MatchHistogram(sampleImage, templImage);
	WriteDiffImage(templImage, sampleImage);
	//sampleImage.convertTo(sampleImage, CV_8U);
	//cv::imwrite("sample.bmp", sampleImage);
#endif

  /*
  // sumTempl (テンプレート画像の画素値の和)を計算する。
	const int numPixels = templImage_.rows * templImage_.cols;
	cv::Mat A = cv::Mat::ones(templImage_.rows, templImage_.cols, CV_8U);
	float sumTempl = static_cast<float>(templImage_.dot(A));
  */

#ifdef DIFF_IS_SUM_SQUARE
	// テンプレート画像の画素値の平均を引くことにより、
	// 黒い部分の多いテンプレートも選ばれやすくなるようにする。
	// 定数は試行錯誤により決める。
  //
  // 係数 < 0.225 (IMG_4488.jpg 右ページ　右から2行目（『望郷の歌』）の'（'を正しく認識するため)
  // 係数 > 0.447 (IMG_4488.jpg 右ページ　右から5行目「少なからざ」の'ざ'を正しく認識するため)
	float ret = sqrt(sumSquare * numPixels) - (sumTempl * 0.223f);

	LogA("%s: avrDiff=%0.3f avrTempl=%0.3f a/b=%0.3f ret=%0.3f\n",
		funcName,
		sqrt(sumSquare / numPixels),                        // avrDiff
		sumTempl / numPixels,                               // avrTempl
		sqrt(sumSquare / numPixels) / sumTempl * numPixels, // a/b
		ret
		);
#else

  // うまくいかないので変更した。
  //
  // 係数 > 0.254 (IMG_4488.jpg 右ページ　右から4行目「あったかが」の「が」を正しく認識するため)
  // 係数 > 0.368 (IMG_4488.jpg 右ページ　右から5行目「少なからざる」の「ざ」を正しく認識するため)
  //float ret = sumSquare - (sumTempl * 0.372f);

  // ○のついているのが正しい答え。
  //
  //'】' MatchingResidue: avrDiff=8.725 avrTempl=38.789 ret=-23365.988
  //'一' MatchingResidue: avrDiff=2.517 avrTempl=16.812 ret=-15308.293 ○
  //
  //'】' MatchingResidue: avrDiff=9.379 avrTempl=38.789 ret=-20687.988
  //'一' MatchingResidue: avrDiff=3.277 avrTempl=16.812 ret=-12193.293 ○
  //
  //'が' MatchingResidue: avrDiff=31.614 avrTempl=71.619 ret=20362.422 ○
  //'か' MatchingResidue: avrDiff=27.390 avrTempl=54.981 ret=28414.109
  //
  //'ざ' MatchingResidue: avrDiff=24.640 avrTempl=58.316 ret=12067.336 ○
  //'さ' MatchingResidue: avrDiff=21.649 avrTempl=50.188 ret=12203.703
  //
  //'▼' MatchingResidue: avrDiff=31.475 avrTempl=97.704 ret=0.322
  //'軍' MatchingResidue: avrDiff=32.678 avrTempl=94.277 ret=0.347 ○
  //
  //'▼' MatchingResidue: avrDiff=32.565 avrTempl=97.704 ret=0.333
  //'軍' MatchingResidue: avrDiff=34.041 avrTempl=94.277 ret=0.361 ○


  // sumSample (サンプル画像の画素値の和)を計算する。
	const int numPixels = sampleImage_.rows * sampleImage_.cols;
	cv::Mat A = cv::Mat::ones(sampleImage_.rows, sampleImage_.cols, CV_8U);
	float sumSample = static_cast<float>(sampleImage_.dot(A));

  float ret = sumSquare / sumSample;

	LogA("%s: limit=%d avrDiff=%0.3f avrSample=%0.3f ret=%0.3f\n",
		funcName,
    limit,
		sumSquare / numPixels,
		sumSample / numPixels,
		ret
		);
#endif

	return ret;
}
