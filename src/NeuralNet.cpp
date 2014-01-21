/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

#include "stdafx.h"
#include "config.h"

#ifdef HAVE_CPPAMP
 #include <amp.h>
 #include <ampblas.h>
#endif

#include "NeuralNet.h"
#include "reshape.h"
#include "log.h"

static GPU_info_type global_GPU_info;

static void printMat(const cv::Mat &m)
{
  if (m.type() != CV_8U && m.type() != CV_32F && m.type() != CV_64F && m.type() != CV_32S)
    return;

  for (int y = 0; y < m.rows; ++y)
  {
    for (int x = 0; x < m.cols; ++x)
    {
      if (m.type() == CV_8U)
        printf("%d ", m.at<unsigned char>(y, x));
      else if (m.type() == CV_32F)
        printf("%f ", m.at<float>(y, x));
      else if (m.type() == CV_64F)
        printf("%lf ", m.at<double>(y, x));
      else if (m.type() == CV_32S)
        printf("%d ", m.at<int>(y, x));
    }
    printf("\n");
  }
}

// element wise multiplication : B_i *= A_i for all i.
static void elemMul(
  const cv::Mat &A_,
  cv::Mat &B_
  )
{
#ifdef _DEBUG
  if (A_.size[0] != B_.size[0] || A_.total() != B_.total())
  {
    printf("elemMul: Matrix sizes are different.\n");
    return;
  }
  if (A_.type() != CV_REAL || B_.type() != CV_REAL)
  {
    printf("elemMul: Supports only %s type.\n", realStr);
    return;
  }
#endif

#ifdef HAVE_CPPAMP

  if (global_GPU_info.GPU_exists)
  {
#ifdef REAL_IS_FLOAT
    const int rowBlockSize = 20000;
#else
    const int rowBlockSize = 10000;
#endif

    const int A_rows = A_.size[0];
    const int A_cols = (int)A_.total() / A_.size[0];
    const int numBlocks = (A_rows - 1)/ rowBlockSize + 1;
    for (int block = 0; block < numBlocks; ++block)
    {
      int rowStart = block * rowBlockSize;
      int numRows = (block < numBlocks - 1)? rowBlockSize : A_rows - (numBlocks - 1) * rowBlockSize;

      concurrency::array_view<const real, 2> A(numRows, A_cols, A_.ptr<real>(rowStart));
      concurrency::array_view<real, 2> B(numRows, A_cols, B_.ptr<real>(rowStart));
      parallel_for_each(A.extent, [=](concurrency::index<2> idx) restrict(amp)
      {
        B[idx] *= A[idx];
      });
      B.synchronize();
    }
  }
  else
    cv::multiply(A_, B_, B_);
#else
  cv::multiply(A_, B_, B_);
#endif
}
/*
static void elemMul(
  const real *pA,
  const int size, 
  real *pB
  )
{
#ifdef HAVE_CPPAMP

  concurrency::array_view<const real, 1> A(size, pA);
  concurrency::array_view<real, 1> B(size, pB);
  parallel_for_each(B.extent, [=](concurrency::index<1> idx) restrict(amp)
  {
    B[idx] *= A[idx];
  });
  B.synchronize();

#else
  const cv::Mat A_(1, size, CV_REAL, const_cast<real *>(pA));  // alias
  cv::Mat B_(1, size, CV_REAL, pB);  // alias
  cv::multiply(A_, B_, B_);
#endif
}
*/

// element wise multiplication : C_i = A_i * B_i for all i.
static void elemMul(
  const cv::Mat &A_,
  const cv::Mat &B_,
  cv::Mat &C_
  )
{
#ifdef _DEBUG
  if (A_.size[0] != B_.size[0] || A_.total() != B_.total())
  {
    printf("elemMul: Matrix sizes are different.\n");
    return;
  }
  if (A_.type() != CV_REAL || B_.type() != CV_REAL)
  {
    printf("elemMul: Supports only %s type.\n", realStr);
    return;
  }
#endif

#ifdef HAVE_CPPAMP

  if (global_GPU_info.GPU_exists)
  {
    const int A_rows = A_.size[0];
    const int A_cols = (int)A_.total() / A_.size[0];

    if (A_rows < 1)
      return;

    if (A_.size[0] != C_.size[0] || A_.total() != C_.total() || A_.type() != C_.type())
      C_.create(A_rows, A_cols, A_.type());

    // rowBlockSize は 20000 で OK, 30000 で NG.
#ifdef REAL_IS_FLOAT
    const int rowBlockSize = 20000;
#else
    const int rowBlockSize = 10000;
#endif

    const int numBlocks = (A_rows - 1)/ rowBlockSize + 1;
    for (int block = 0; block < numBlocks; ++block)
    {
      int rowStart = block * rowBlockSize;
      int numRows = (block < numBlocks - 1)? rowBlockSize : A_rows - (numBlocks - 1) * rowBlockSize;

      concurrency::array_view<const real, 2> A(numRows, A_cols, A_.ptr<real>(rowStart));
      concurrency::array_view<const real, 2> B(numRows, A_cols, B_.ptr<real>(rowStart));
      concurrency::array_view<real, 2> C(numRows, A_cols, C_.ptr<real>(rowStart));
      C.discard_data();
      parallel_for_each(A.extent, [=](concurrency::index<2> idx) restrict(amp)
      {
        C[idx] = A[idx] * B[idx];
      });
      C.synchronize();
    }
  }
  else
    cv::multiply(A_, B_, C_);

#else
  cv::multiply(A_, B_, C_);
#endif
}
/*
static void elemMul(
  const real *pA,
  const real *pB,
  const int size, 
  real *pC
  )
{
#ifdef HAVE_CPPAMP

  concurrency::array_view<const real, 1> A(size, pA);
  concurrency::array_view<const real, 1> B(size, pB);
  concurrency::array_view<real, 1> C(size, pC);
  C.discard_data();
  parallel_for_each(B.extent, [=](concurrency::index<1> idx) restrict(amp)
  {
    C[idx] = A[idx] * B[idx];
  });
  C.synchronize();

#else
  const cv::Mat A(1, size, CV_REAL, const_cast<real *>(pA));  // alias
  const cv::Mat B(1, size, CV_REAL, const_cast<real *>(pB));  // alias
  cv::Mat C(1, size, CV_REAL, pC);  // alias
  cv::multiply(A, B, C);
#endif
}
*/

static void matMul(
  const cv::Mat &src1,
  const cv::Mat &src2,
  real alpha,
  cv::Mat &dst,
  int flags = 0)
{
  if (
#ifdef HAVE_CPPAMP
    true
#else
    false
#endif
    && global_GPU_info.GPU_exists &&
    (global_GPU_info.supportsDouble || 
#ifdef REAL_IS_FLOAT
    true
#else
    false
#endif
    ))
  {
    //std::cout << "flags: " << flags << std::endl;
    //std::cout << "src1 (" << src1.rows << "x" << src1.cols << ") : " << std::endl;
    //printMat(src1);
    //std::cout << "src2 (" << src2.rows << "x" << src2.cols << ") : " << std::endl;
    //printMat(src2);

    cv::Mat src1_;
    if (flags & cv::GEMM_1_T)
      src1_ = src1.t();
    else
      src1_ = src1;

    CV_Assert(src1_.isContinuous() && src2.isContinuous());
    CV_Assert(src1_.cols == ((flags & cv::GEMM_2_T)? src2.cols : src2.rows));

    auto accView = concurrency::accelerator().default_view;
    const int dstRows = (flags & cv::GEMM_1_T)? src1.cols : src1.rows;
    const int dstCols = (flags & cv::GEMM_2_T)? src2.rows : src2.cols;
    if (dst.rows != dstRows || dst.cols != dstCols || dst.type() != CV_REAL)
      dst.create(dstRows, dstCols, CV_REAL);

    const int rowBlockSize = 20000;
    const int numBlocks = (src1_.rows - 1)/ rowBlockSize + 1;
    for (int block = 0; block < numBlocks; ++block)
    {
      int rowStart = block * rowBlockSize;
      int numRows = (block < numBlocks - 1)? rowBlockSize : src1_.rows - (numBlocks - 1) * rowBlockSize;

      concurrency::array_view<const real, 2> src1_view(numRows, src1_.cols, src1_.ptr<real>(rowStart, 0));
      concurrency::array_view<const real, 2> src2_view(src2.rows, src2.cols, (real *)(src2.data));
      concurrency::array_view<real, 2> dst_view(numRows, dstCols, dst.ptr<real>(rowStart, 0));
      dst_view.discard_data();

      // This function takes column-major matrices.
      // We just have to switch src1 and src2.
      ampblas::gemm<real>(
        accView,
        (flags & cv::GEMM_2_T)? ampblas::transpose::trans : ampblas::transpose::no_trans,
        ampblas::transpose::no_trans,
        alpha,
        src2_view,
        src1_view,
        0.0,
        dst_view);
      dst_view.synchronize();
    }

    if (flags & cv::GEMM_3_T)
      dst = dst.t();
  }
  else
    cv::gemm(src1, src2, alpha, 0, 0, dst, flags);

  //std::cout << "dst (" << dst.rows << "x" << dst.cols << ") : " << std::endl;
  //printMat(dst);
}
/*
static void matMul(
  const cv::Mat &src1,
  const cv::Mat &src2,
  real alpha,
  cv::Mat &dst,
  int flags = 0)
{
#ifdef HAVE_CPPAMP
  auto accView = concurrency::accelerator().default_view;
  const int dstRows = (flags & cv::GEMM_1_T)? src1.cols : src1.rows;
  const int dstCols = (flags & cv::GEMM_2_T)? src2.rows : src2.cols;
  if (dst.rows != dstRows || dst.cols != dstCols)
    dst.create(dstRows, dstCols, CV_REAL);
  concurrency::array_view<const real, 2> src1_view(src1.rows, src1.cols, (real *)(src1.data));
  concurrency::array_view<const real, 2> src2_view(src2.rows, src2.cols, (real *)(src2.data));
  concurrency::array_view<real, 2> dst_view(dstRows, dstCols, (real *)(dst.data));
  dst_view.discard_data();

  // This function takes column-major matrices.
  // dst^T <- src2^T * src1^T
  ampblas::gemm<real>(
    accView,
    (flags & cv::GEMM_2_T)? ampblas::transpose::trans : ampblas::transpose::no_trans,
    (flags & cv::GEMM_1_T)? ampblas::transpose::trans : ampblas::transpose::no_trans,
    alpha,
    src2_view,
    src1_view,
    0.0,
    dst_view);
  dst_view.synchronize();
#else
  cv::gemm(src1, src2, alpha, 0, 0, dst, flags);
#endif
}
*/

// class NNLayer --------------------------------------------------------------------------------

void NNLayer::logSettings() const
{
  switch (type)
  {
  case perceptron:
    Log("Tanh perceptron layer : %d -> %d, dropout ratio = %f\n", inSize, outSize, (float)dropoutRatio);
    break;
  case linearPerceptron:
    Log("Linear perceptron layer : %d -> %d, dropout ratio = %f, max weight norm = %f\n",
      inSize, outSize, (float)dropoutRatio, (float)maxWeightNorm);
    break;
  case convolution:
    Log("Convolution layer : %d@%dx%d -> %d@%dx%d (filter %dx%d), dropout ratio = %f\n",
      numInMaps, inMapSize.at<int>(0), inMapSize.at<int>(1),
      numOutMaps, outMapSize.at<int>(0), outMapSize.at<int>(1),
      filterSize.at<int>(0), filterSize.at<int>(1),
      (float)dropoutRatio);
    break;
  case maxPool:
    Log("Maxpool layer : filter %dx%d\n", filterSize.at<int>(0), filterSize.at<int>(1));
    break;
  case softMax:
    Log("Softmax layer : %d -> %d, \n", inSize, outSize);
    break;
  default:
    Log("Unknown type of layer : %d\n", type);
  }
}

void NNLayer::createPerceptronLayer(const int inSize_, const int outSize_, const real dropoutRatio_)
{
  type = perceptron;
  inSize = inSize_;
  outSize = outSize_;
  dropoutRatio = dropoutRatio_;
  weight.create(inSize, outSize, CV_REAL);
  bias.create(1, outSize, CV_REAL);
}

void NNLayer::createLinearPerceptronLayer(
  const int inSize_,
  const int outSize_,
  const real dropoutRatio_,
  const real maxWeightNorm_)
{
  type = linearPerceptron;
  inSize = inSize_;
  outSize = outSize_;
  dropoutRatio = dropoutRatio_;
  maxWeightNorm = maxWeightNorm_;
  weight.create(inSize, outSize, CV_REAL);
  bias.create(1, outSize, CV_REAL);
}

void NNLayer::createSoftMaxLayer(const int inOutSize)
{
  type = softMax;
  inSize = outSize = inOutSize;
  //weight.create(inSize, outSize, CV_REAL);
  //bias.create(1, outSize, CV_REAL);
}

void NNLayer::createConvolutionLayer(
  const cv::Mat inMapSize_,   // e.g. {13, 13} for input feature maps of size HxW=13x13
  const cv::Mat filterSize_,  // e.g. {5, 5} for HxW=5x5 filters.
  const int numInMaps_,       // number of input feature maps
  const int numOutMaps_,      // number of putput feature maps
  const real dropoutRatio_
  )
{
  type = convolution;
  inMapSize = inMapSize_;
  filterSize = filterSize_;
  outMapSize = inMapSize - filterSize + 1;
  numInMaps = numInMaps_;
  numOutMaps = numOutMaps_;
  dropoutRatio = dropoutRatio_;

  // weight is <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
  // e.g. when numOutMaps=40, numInMaps=20 and filterSize=5x5,
  // weight shall be 40x20x5x5 matrix.
  cv::Mat weightSize = (cv::Mat_<int>(1, 4) << numOutMaps, numInMaps, filterSize.at<int>(0), filterSize.at<int>(1));
  weight.create(4, (int *)(weightSize.data), CV_REAL);
  bias.create(1, numOutMaps, CV_REAL);
}

void NNLayer::createMaxPoolLayer(
  const cv::Mat filterSize_  // e.g. {2, 2} for HxW=2x2 filters. 
  )
{
  type = maxPool;
  filterSize = filterSize_;
}

// Calculate activation for perceptron or convolutional layers.
//
// calculate:
//
// activation = 1.7159 * tanh(y * 2 / 3)
//
// df = 1.7159 * 2 / 3 * (1 - tanh(y * 2 / 3)^2);
void NNLayer::activateTanh(const cv::Mat &y)
{
  // C++ AMP : Windows 7 では double 型は制限つきサポートになる。
  // ・ concurrency::precise_math の関数は使用できない。
  // ・割り算もできない。
  //
  // http://blogs.msdn.com/b/nativeconcurrency/archive/2012/02/07/double-precision-support-in-c-amp.aspx
  //

  if (
#ifdef HAVE_CPPAMP
    true
#else
    false
#endif
    && global_GPU_info.GPU_exists &&
    (global_GPU_info.supportsDouble || 
#ifdef REAL_IS_FLOAT
    true
#else
    false
#endif
    ))
  {
    //std::cout << "y : " << y << std::endl;

    const int rowBlockSize = 20000;
    const int y_rows = y.size[0];
    const int y_cols = (int)y.total() / y.size[0];
    const int numBlocks = (y_rows - 1)/ rowBlockSize + 1;
    for (int block = 0; block < numBlocks; ++block)
    {
      int rowStart = block * rowBlockSize;
      int numRows = (block < numBlocks - 1)? rowBlockSize : y_rows - (numBlocks - 1) * rowBlockSize;

      concurrency::array_view<const real, 2> y_(numRows, y_cols, y.ptr<real>(rowStart, 0));
      concurrency::array_view<real, 2> activation_(numRows, y_cols, activation.ptr<real>(rowStart, 0));
      concurrency::array_view<real, 2> df_(numRows, y_cols, df.ptr<real>(rowStart, 0));
      activation_.discard_data();
      df_.discard_data();
      parallel_for_each(y_.extent, [=](concurrency::index<2> idx) restrict(amp)
      {
        real th;
#ifdef REAL_IS_FLOAT
        if (y_[idx] < -100)
          th = -1;
        else if (100 < y_[idx])
          th = 1;
        else
          th = concurrency::fast_math::tanh(y_[idx] * 2 / 3);
#else
        if (y_[idx] < -100)
          th = -1;
        else if (100 < y_[idx])
          th = 1;
        else
          th = concurrency::precise_math::tanh(y_[idx] * 2 / 3);
#endif
        activation_[idx] = real(1.7159) * th;
        df_[idx] = real(1.7159) * 2 * (real(1.0) - th * th) / 3;
      });
      activation_.synchronize();
      df_.synchronize();
    }
    //std::cout << "activation : " << activation << std::endl;
    //std::cout << "df : " << df << std::endl;
  }
  else
  {
    cv::Mat exp_y = (real(4) / 3) * y;
    cv::exp(exp_y, exp_y);
  
    //workaround for : activation = 1 - 2 / (exp_y + 1);
#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)exp_y.total(); ++i)
      *(real *)(activation.data + i * sizeof(real)) = 1 - 2 / (*(real *)(exp_y.data + i * sizeof(real)) + 1);

    df = (real(1.7159) * 2 / 3) * (1 - activation.mul(activation));
    activation *= real(1.7159);

    /*
    for (int s = 0; s < y.rows; ++s)
    {
      for (int i = 0; i < y.cols; ++i)
      {
        real th = tanh(real(2) / 3 * y.at<real>(s, i));
        activation.at<real>(s, i) = real(1.7159) * th;
        df.at<real>(s, i) = real(1.7159) * 2 / 3 * (real(1.0) - th * th);
      }
    }
    */
  }
}

// Calculate activation for softmax layers.
//
// calculate:
//   activation_i = exp(y_i)/Sigma_k(exp(y_k))
//
void NNLayer::activateSoftMax(const cv::Mat &y_)
{
  //std::cout << "activateSoftMax(): y_ : " << std::endl;
  //printMat(y_);

  const int numSamples = y_.rows;
  cv::Mat y;

#ifdef REAL_IS_FLOAT
  
  // 大きい数（例えば 90）を exp すると 1.#INF00 になってしまうので
  // 各サンプル（＝列）の最大値が 0 となるよう、それぞれの行から定数を引く。
  cv::Mat rowsMax;
  cv::reduce(y_, rowsMax, 1, CV_REDUCE_MAX);  // calculate max of each row
  y = y_.clone();

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
  for (int s = 0; s < numSamples; ++s)
    y.row(s) -= rowsMax.at<real>(s);

#else
    y = y_;
#endif

  //std::cout << "activateSoftMax(): y : " << std::endl;
  //printMat(y);

  cv::Mat exp_y;
  cv::exp(y, exp_y);

  //std::cout << "activateSoftMax(): exp_y : " << std::endl;
  //printMat(exp_y);

  cv::Mat sum_exp = exp_y * cv::Mat::ones(outSize, 1, CV_REAL);

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
  for (int s = 0; s < numSamples; ++s)
    exp_y.row(s) /= sum_exp.at<real>(s);

  activation = exp_y;

  //std::cout << "activateSoftMax(): activation : " << std::endl;
  //printMat(activation);
}

// get a random permutation of [0, 1, ..., len-1] where len = vec.size().
static void GetRandomPermutation(std::vector<unsigned int> &vec, cv::RNG &rng)
{
  // Knuth shuffles algorithm
  const size_t len = vec.size();
  for (unsigned int i = 0; i < len; ++i)
  {
    unsigned int j = rng.uniform(0, i + 1);
    vec[i] = vec[j];
    vec[j] = i;
  }
}

// For each row in mask, 'dropoutRatio' by ratio of columns shall be set to zero.
static void makeDropoutMask(cv::Mat &mask, const real &dropoutRatio, cv::RNG &rng)
{
  const int numSamples = mask.size[0];
  const int sampleSize = (int)mask.total() / numSamples;
  std::vector<unsigned int> vec(sampleSize);

  for (int s = 0; s < numSamples; ++s)
  {
    GetRandomPermutation(vec, rng);
    cv::Mat randPermu(1, sampleSize, CV_32S, &vec[0]);
    cv::Mat aRow(1, sampleSize, mask.type(), mask.ptr(s));
    aRow = (randPermu >= (unsigned int)(sampleSize * dropoutRatio));
  }
}

// For each sample (=row) in the input, 'dropoutRatio' by ratio of columns shall be set to zero.
static void Dropout(const cv::Mat &inputs, cv::Mat &dropped, const real &dropoutRatio)
{
  //cv::Mat mask(inputs.size(), CV_8U);
  cv::Mat mask(inputs.dims, inputs.size, CV_8U);
	cv::RNG rng(time(NULL));
  makeDropoutMask(mask, dropoutRatio, rng);

  dropped = 0;
  inputs.copyTo(dropped, mask);
}

void NNLayer::forwardPropagate(const cv::Mat* &pX, const bool dropout, const bool copyBackDropout)
{
  if (type == perceptron || type == linearPerceptron)
  {
    const int numSamples = pX->size[0];
    const int x_cols = (int)(pX->total()) / pX->size[0];
    CV_Assert(x_cols == inSize);

    cv::Mat x;
    x = cv::Mat(numSamples, x_cols, CV_REAL, pX->data);
    pX = &x;

    cv::Mat dropped;
    if (dropoutRatio != 0)
    {
      if (dropout)
        Dropout(x, dropped, dropoutRatio);
      else
        dropped = x * (1 - dropoutRatio);
      pX = &dropped;
      if (copyBackDropout)
        dropped.copyTo(x);
    }

    if (type == perceptron)
    {
      cv::Mat y(numSamples, outSize, CV_REAL);
      // y <- x * weight
      //std::cout << "*pX : " << pX->row(0) << std::endl;
      //std::cout << "weight : " << weight << std::endl;
      matMul(*pX, weight, 1, y);
      //std::cout << "y : " << y.row(0) << std::endl;

      // activation += bias
      for (int r = 0; r < numSamples; ++r)
        y.row(r) += bias;

      // calculate df and activation, i.e.,
      // df <- f'(y),
      // activation <- f(y).
      activateTanh(y);
    }
    else if (type == linearPerceptron)
    {
      // activation <- x * weight
      //std::cout << std::endl << "forwardPropagate(): *pX : " << *pX << std::endl;
      //std::cout << std::endl << "forwardPropagate(): weight : " << weight << std::endl;
      matMul(*pX, weight, (dropout? 1 : (1 - dropoutRatio)), activation);
      //std::cout << std::endl << "forwardPropagate(): activation after matMul: " << activation << std::endl;

      // activation += bias
      for (int r = 0; r < numSamples; ++r)
        activation.row(r) += bias;

      //std::cout << std::endl << "forwardPropagate(): activation after adding bias : " << activation << std::endl;
    }
  }
  else if (type == softMax)
  {
    const int numSamples = pX->size[0];
    const int x_cols = (int)(pX->total()) / pX->size[0];
    CV_Assert(x_cols == inSize);

    cv::Mat y(numSamples, x_cols, CV_REAL, pX->data);

    // calculate only activation 
    activateSoftMax(y);
  }
  else if (type == maxPool)
  {
    cv::Mat x;

    // *pX is either
    // <num samples>x<numInMaps>x<in_H>x<in_W> matrix, or
    // <num samples>x<in_Hxin_W> matrix. (numInMaps should be 1)
    const int numSamples = pX->size[0];
    if (pX->dims == 4)
    {
      CV_Assert(pX->size[1] == numInMaps);
      CV_Assert(pX->size[2] == inMapSize.at<int>(0));
      CV_Assert(pX->size[3] == inMapSize.at<int>(1));
    }
    else
    {
      CV_Assert(pX->dims == 2);
      CV_Assert(pX->size[1] == inMapSize.at<int>(1));

      cv::Mat inMatSize = (cv::Mat_<int>(1, 4) << numSamples, numInMaps, inMapSize.at<int>(0), inMapSize.at<int>(1));
      //precLayer.grad = reshape(precLayer.grad, 1, 4, (int *)(inMatSize.data));
      x = cv::Mat(4, (int *)inMatSize.data, CV_REAL, pX->data);
      pX = &x;
    }

    // Now, *pX is <num samples>x<numInMaps>x<in_H>x<in_W> matrix.
    //
    // activation and df are <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
    const int in_H = inMapSize.at<int>(0);
    const int in_W = inMapSize.at<int>(1);
    const int out_H = outMapSize.at<int>(0);
    const int out_W = outMapSize.at<int>(1);
    const int filter_H = filterSize.at<int>(0);
    const int filter_W = filterSize.at<int>(1);

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
    for (int s = 0; s < numSamples; ++s)
    {
      for (int map = 0; map < numInMaps; ++map)
      {
        cv::Mat currentInMap(in_H, in_W, pX->type(), pX->data + pX->step[0] * s + pX->step[1] * map);
        cv::Mat currentActivation(out_H, out_W, CV_REAL, activation.data + activation.step[0] * s + activation.step[1] * map);
        cv::Mat currentDf(out_H, out_W, CV_32S, df.data + df.step[0] * s + df.step[1] * map);

        for (int y = 0; y < out_H; ++y)
        {
          for (int x = 0; x < out_W; ++x)
          {
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(currentInMap(cv::Rect(x * filter_W, y * filter_H, filter_W, filter_H)), &minVal, &maxVal, &minLoc, &maxLoc);
            currentActivation.at<real>(y, x) = (real)maxVal;
            currentDf.at<_int32>(y, x) = maxLoc.x + maxLoc.y * filter_W;
          }
        }
      }
    }
    //cv::Mat tmp(df.size[0], df.total() / df.size[0], df.type(), df.data);
    //std::cout << std::endl << "df : " << std::endl;
    //printMat(tmp);
  }
  else if (type == convolution)
  {
    cv::Mat x;

    // *pX is either
    // <num samples>x<numInMaps>x<in_H>x<in_W> matrix, or
    // <num samples>x<in_Hxin_W> matrix. (numInMaps should be 1)
    const int numSamples = pX->size[0];
    if (pX->dims == 4)
    {
      CV_Assert(pX->size[1] == numInMaps);
      CV_Assert(pX->size[2] == inMapSize.at<int>(0));
      CV_Assert(pX->size[3] == inMapSize.at<int>(1));
      x = *pX;
    }
    else
    {
      CV_Assert(pX->dims == 2);
      CV_Assert(numInMaps == 1);
      CV_Assert(pX->size[1] == inMapSize.at<int>(0) * inMapSize.at<int>(1));

      cv::Mat inMatSize = (cv::Mat_<int>(1, 4) << numSamples, numInMaps, inMapSize.at<int>(0), inMapSize.at<int>(1));
      //precLayer.grad = reshape(precLayer.grad, 1, 4, (int *)(inMatSize.data));
      x = cv::Mat(4, (int *)inMatSize.data, CV_REAL, pX->data);
      pX = &x;
    }

    // Now, *pX is <num samples>x<numInMaps>x<in_H>x<in_W> matrix.

    cv::Mat dropped;
    if (dropoutRatio != 0)
    {
      if (dropout)
        Dropout(x, dropped, dropoutRatio);
      else
        dropped = x * (1 - dropoutRatio);
      pX = &dropped;
      if (copyBackDropout)
        dropped.copyTo(x);
    }

    // y is <num samples>x<numOutMaps>x<out_H>x<out_W> matrix
    cv::Mat matSize = (cv::Mat_<int>(1, 4) << numSamples, numOutMaps, outMapSize.at<int>(0), outMapSize.at<int>(1));
    cv::Mat y(4, (int *)matSize.data, CV_REAL);
    CV_Assert(y.data != 0);

    const int in_H = inMapSize.at<int>(0);
    const int in_W = inMapSize.at<int>(1);
    const int out_H = outMapSize.at<int>(0);
    const int out_W = outMapSize.at<int>(1);
    const int filter_H = filterSize.at<int>(0);
    const int filter_W = filterSize.at<int>(1);

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
    for (int s = 0; s < numSamples; ++s)
    {
      for (int outMap = 0; outMap < numOutMaps; ++outMap)
      {
        cv::Mat outImage(out_H, out_W, CV_REAL, y.data + y.step[0] * s + y.step[1] * outMap);
        outImage = bias.at<real>(outMap);

        for (int inMap = 0; inMap < numInMaps; ++inMap)
        {
          cv::Mat inImage(in_H, in_W, CV_REAL, pX->data + pX->step[0] * s + pX->step[1] * inMap);

          // weight is <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
          cv::Mat weightSub(filter_H, filter_W, CV_REAL, weight.data + weight.step[0] * outMap + weight.step[1] * inMap);
          cv::Ptr<cv::FilterEngine> pEngine = createLinearFilter(CV_REAL, CV_REAL, weightSub);
          cv::Mat dst(in_H, in_W, CV_REAL);
          pEngine->apply(inImage, dst);
          if (!dropout && dropoutRatio != 0)
            dst *= (1 - dropoutRatio);
          outImage += dst(cv::Rect((filter_W - 1) / 2, (filter_H - 1) / 2, out_W, out_H));
        }
      }
    }

    // calculate df and activation, i.e.,
    // df <- f'(y),
    // activation <- f(y).
    activateTanh(y);
  }

  //std::cout << "activation : " << activation.row(0) << std::endl;
  pX = &activation;
}

void NNLayer::UpdateWeightsRprop(const cv::Mat &dEdw, const updateParam &update_param)
{

  if (
#ifdef HAVE_CPPAMP
    true
#else
    false
#endif
    && global_GPU_info.GPU_exists &&
    (global_GPU_info.supportsDouble || 
#ifdef REAL_IS_FLOAT
    true
#else
    false
#endif
    ))
  {
    const int w_rows = weight.size[0];
    const int w_cols = (int)weight.total() / weight.size[0];
    const int rowBlockSize = 20000;
    const int numBlocks = (w_rows - 1)/ rowBlockSize + 1;
    for (int block = 0; block < numBlocks; ++block)
    {
      int rowStart = block * rowBlockSize;
      int numRows = (block < numBlocks - 1)? rowBlockSize : w_rows - (numBlocks - 1) * rowBlockSize;

      concurrency::array_view<const real, 2> dEdw_  (numRows, w_cols, dEdw.ptr<real>(rowStart, 0));
      concurrency::array_view<real      , 2> dw_    (numRows, w_cols, dw.ptr<real>(rowStart, 0));
      concurrency::array_view<real      , 2> weight_(numRows, w_cols, weight.ptr<real>(rowStart, 0));
      concurrency::array_view<int       , 2> dwSign_(numRows, w_cols, dwSign.ptr<int>(rowStart, 0));

      parallel_for_each(weight_.extent, [=](concurrency::index<2> idx) restrict(amp)
      {
        int sign = -CV_SIGN(dEdw_[idx]);
        int ss = sign * dwSign_[idx];
        if (ss > 0)
        {
          real dval = dw_[idx] * update_param.dw_plus;
          if (dval > update_param.dw_max)
            dval = update_param.dw_max;
          dw_[idx] = dval;
          weight_[idx] += dval * sign;
        }
        else if (ss < 0)
        {
          real dval = dw_[idx] * update_param.dw_minus;
          if (dval < update_param.dw_min)
            dval = update_param.dw_min;
          dwSign_[idx] = 0;
          dw_[idx] = dval;
          weight_[idx] += dval * sign;
        }
        else
        {
          dwSign_[idx] = sign;
          weight_[idx] += dw_[idx] * sign;
        }
      });
      weight_.synchronize();
      dw_.synchronize();
      dwSign_.synchronize();
    }
  }
  else
  {
    //int numWeights = inSize * outSize;
    int numWeights = (int)weight.total();

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
    for (int j = 0; j < numWeights; ++j)
    {
      int sign = -CV_SIGN(((real *)dEdw.data)[j]);
      int ss = sign * ((char *)dwSign.data)[j];
      if (ss > 0)
      {
        real dval = ((real *)dw.data)[j] * update_param.dw_plus;
        dval = std::min(dval, update_param.dw_max);
        ((real *)dw.data)[j] = dval;
        ((real *)weight.data)[j] += dval * sign;
      }
      else if (ss < 0)
      {
        real dval = ((real *)dw.data)[j] * update_param.dw_minus;
        dval = std::max(dval, update_param.dw_min);
        ((char *)dwSign.data)[j] = 0;
        ((real *)dw.data)[j] = dval;
        ((real *)weight.data)[j] += dval * sign;
      }
      else
      {
        ((char *)dwSign.data)[j] = (char)sign;
        ((real *)weight.data)[j] += ((real *)dw.data)[j] * sign;
      }
    }
  }
}

void NNLayer::writeBinary(FILE *fp) const
{
  fwrite(&type, sizeof(type), 1, fp);
  if (type == perceptron || type == linearPerceptron)
  {
    fwrite(&inSize, sizeof(inSize), 1, fp);
    fwrite(&outSize, sizeof(outSize), 1, fp);
    fwrite(weight.data, sizeof(real), weight.cols * weight.rows, fp);
    fwrite(bias.data, sizeof(real), bias.cols, fp);
    fwrite(&dropoutRatio, sizeof(dropoutRatio), 1, fp);
  }
  else if (type == softMax)
  {
    fwrite(&inSize, sizeof(inSize), 1, fp);
  }
  else if (type == convolution)
  {
    fwrite(inMapSize.data, sizeof(int), 2, fp);
    fwrite(filterSize.data, sizeof(int), 2, fp);
    fwrite(&numInMaps, sizeof(numInMaps), 1, fp);
    fwrite(&numOutMaps, sizeof(numOutMaps), 1, fp);
    fwrite(&dropoutRatio, sizeof(dropoutRatio), 1, fp);

    // weight is <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
    fwrite(weight.data, sizeof(real), weight.total(), fp);

    // bias is 1x<numOutMaps> matrix.
    fwrite(bias.data, sizeof(real), bias.cols, fp);
  }
  else if (type == maxPool)
  {
    fwrite(inMapSize.data, sizeof(int), 2, fp);
    fwrite(filterSize.data, sizeof(int), 2, fp);
    fwrite(&numInMaps, sizeof(numInMaps), 1, fp);
  }
}

void NNLayer::readBinary(FILE *fp)
{
  fread(&type, sizeof(type), 1, fp);
  if (type == perceptron || type == linearPerceptron)
  {
    fread(&inSize, sizeof(inSize), 1, fp);
    fread(&outSize, sizeof(outSize), 1, fp);

    weight.create(inSize, outSize, CV_REAL);
    fread(weight.data, sizeof(real), weight.cols * weight.rows, fp);

    bias.create(1, outSize, CV_REAL);
    fread(bias.data, sizeof(real), bias.cols, fp);

    fread(&dropoutRatio, sizeof(dropoutRatio), 1, fp);
  }
  else if (type == softMax)
  {
    fread(&inSize, sizeof(inSize), 1, fp);
    outSize = inSize;
  }
  else if (type == convolution)
  {
    inMapSize.create(1, 2, CV_32S);
    fread(inMapSize.data, sizeof(int), 2, fp);
    filterSize.create(1, 2, CV_32S);
    fread(filterSize.data, sizeof(int), 2, fp);
    outMapSize = inMapSize - filterSize + 1;
    fread(&numInMaps, sizeof(int), 1, fp);
    fread(&numOutMaps, sizeof(int), 1, fp);
    fread(&dropoutRatio, sizeof(dropoutRatio), 1, fp);

    // weight is <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
    cv::Mat weightSize = (cv::Mat_<int>(1, 4) << numOutMaps, numInMaps, filterSize.at<int>(0), filterSize.at<int>(1));
    weight.create(4, (int *)(weightSize.data), CV_REAL);
    fread(weight.data, sizeof(real), weight.total(), fp);

    // bias is 1x<numOutMaps> matrix.
    bias.create(1, numOutMaps, CV_REAL);
    fread(bias.data, sizeof(real), bias.cols, fp);
  }
  else if (type == maxPool)
  {
    inMapSize.create(1, 2, CV_32S);
    fread(inMapSize.data, sizeof(int), 2, fp);
    filterSize.create(1, 2, CV_32S);
    fread(filterSize.data, sizeof(int), 2, fp);
    outMapSize = inMapSize / filterSize;
    fread(&numInMaps, sizeof(int), 1, fp);
    numOutMaps = numInMaps;
  }
}

// class NeuralNet --------------------------------------------------------------------------------

NeuralNet::NeuralNet() : firstCharCode(0)
{
#ifdef HAVE_CPPAMP

  concurrency::accelerator def;

  GPU_info.description = def.description;
  GPU_info.device_path = def.device_path;
  LogW(L"accelerator: %s - %s\n", GPU_info.description.data(), GPU_info.device_path.data());
  wprintf(L"accelerator: %s - %s\n", GPU_info.description.data(), GPU_info.device_path.data());

  GPU_info.GPU_exists = (GPU_info.description.find(L"Software Adapter") == std::wstring::npos)? true : false;

  GPU_info.limitedDouble = def.supports_limited_double_precision;
  GPU_info.supportsDouble = def.supports_double_precision;
  printf("limited double precision : %s\n", GPU_info.limitedDouble? "true" : "false");
  Log("limited double precision : %s\n", GPU_info.limitedDouble? "true" : "false");
  printf("supports double precision : %s\n", GPU_info.supportsDouble? "true" : "false");
  Log("supports double precision : %s\n", GPU_info.supportsDouble? "true" : "false");

  global_GPU_info = GPU_info;

#endif
}

void NeuralNet::logSettings() const
{
  Log("Layer settings:\n");
  for (int i = 0; i < (int)layers.size(); ++i)
  {
    Log("  ");
    layers[i].logSettings();
  }
}

int NeuralNet::constructLayers(const std::string &layerParamStr)
{
  std::istringstream iss(layerParamStr);
  std::string segment;
  int numLayers = 0;
	while (getline(iss, segment, '_'))
    ++numLayers;

  layers.resize(numLayers);
  int currentLayer = 0;

  iss = std::istringstream(layerParamStr);
	while (getline(iss, segment, '_'))
  {
    std::istringstream seg(segment);
    std::string token;
  	getline(seg, token, ',');
    if (token == "P") // tanh perceptron layer
    {
    	if (!getline(seg, token, ','))
        return 1;
      int inSize = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int outSize = atoi(token.c_str());
      real dropoutRatio = 0;
    	if (getline(seg, token, ','))
        dropoutRatio = (real)atof(token.c_str());
      layers[currentLayer++].createPerceptronLayer(inSize, outSize, dropoutRatio);
    }
    else if (token == "L") // linear perceptron layer
    {
    	if (!getline(seg, token, ','))
        return 1;
      int inSize = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int outSize = atoi(token.c_str());
      real dropoutRatio = 0;
      real maxWeightNorm = 0;
    	if (getline(seg, token, ','))
      {
        dropoutRatio = (real)atof(token.c_str());
      	if (getline(seg, token, ','))
          maxWeightNorm = (real)atof(token.c_str());
      }
      layers[currentLayer++].createLinearPerceptronLayer(inSize, outSize, dropoutRatio, maxWeightNorm);
    }
    else if (token == "C") // tanh convolution layer
    {
    	if (!getline(seg, token, ','))
        return 1;
      int inMapH = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int inMapW = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int filterH = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int filterW = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int numInMaps = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int numOutMaps = atoi(token.c_str());
      real dropoutRatio = 0;
    	if (getline(seg, token, ','))
        dropoutRatio = (real)atof(token.c_str());
      layers[currentLayer++].createConvolutionLayer(
        (cv::Mat_<_int32>(1,2) << inMapH, inMapW),
        (cv::Mat_<_int32>(1,2) << filterH, filterW),
        numInMaps, numOutMaps, dropoutRatio);
    }
    else if (token == "M") // maxpool layer
    {
    	if (!getline(seg, token, ','))
        return 1;
      int filterH = atoi(token.c_str());
    	if (!getline(seg, token, ','))
        return 1;
      int filterW = atoi(token.c_str());
      layers[currentLayer++].createMaxPoolLayer((cv::Mat_<_int32>(1,2) << filterH, filterW));
    }
    else if (token == "S") // softmax layer
    {
    	if (!getline(seg, token, ','))
        return 1;
      int inOutSize = atoi(token.c_str());
      layers[currentLayer++].createSoftMaxLayer(inOutSize);
    }
  }

  return 0;
}

int NeuralNet::create(const std::string &layerParamStr)
{
  int ret = constructLayers(layerParamStr);
  if (ret != 0)
    return 100 + ret;

  if (layers[0].type == NNLayer::convolution)
  {
    if (layers[0].numInMaps != 1)
      return 1;
  }
  else if (layers[0].type == NNLayer::maxPool)
  {
    layers[0].numInMaps = layers[0].numOutMaps = 1;
    layers[0].inMapSize = (cv::Mat_<int>(1,2) << 1, 0);
    layers[0].outMapSize = (cv::Mat_<int>(1,2) << 1, 0);
  }

  for (int i = 1; i < numLayers(); ++i)
  {
    if (layers[i - 1].type == NNLayer::perceptron || layers[i - 1].type == NNLayer::linearPerceptron)
    {
      if (
        layers[i].type == NNLayer::perceptron ||
        layers[i].type == NNLayer::linearPerceptron ||
        layers[i].type == NNLayer::softMax)
      {
        if (layers[i - 1].outSize != layers[i].inSize)
        {
          // Size mismatch.
          return 2;
        }
      }
      else if (layers[i].type == NNLayer::convolution)
      {
        if (layers[i].numInMaps != 1)
        {
          return 3;
        }
        else if (layers[i - 1].outSize != layers[i].numOutMaps * layers[i].outMapSize.at<int>(0) * layers[i].outMapSize.at<int>(1))
        {
          // Size mismatch.
          return 4;
        }
      }
      else if (layers[i].type == NNLayer::maxPool)
      {
        layers[i].numInMaps = layers[i].numOutMaps = 1;
        layers[i].inMapSize = (cv::Mat_<int>(1,2) << 1, layers[i - 1].outSize);
        layers[i].outMapSize = (cv::Mat_<int>(1,2) << 1, (layers[i - 1].outSize / layers[i].filterSize.at<int>(1)));
        if (layers[i - 1].outSize % layers[i].filterSize.at<int>(1) != 0)
          return 14;
      }
    }
    else if (layers[i - 1].type == NNLayer::convolution)
    {
      if (layers[i].type == NNLayer::perceptron || layers[i].type == NNLayer::softMax)
      {
        if (layers[i - 1].outMapSize.at<int>(0) * layers[i - 1].outMapSize.at<int>(1) * layers[i - 1].numOutMaps != layers[i].inSize)
        {
          // Size mismatch.
          return 6;
        }
      }
      else if (layers[i].type == NNLayer::convolution)
      {
        if (layers[i - 1].numOutMaps != layers[i].numInMaps)
        {
          // number of feature maps mismatch.
          return 7;
        }
        else if (
          layers[i - 1].outMapSize.at<int>(0) != layers[i].inMapSize.at<int>(0) || 
          layers[i - 1].outMapSize.at<int>(1) != layers[i].inMapSize.at<int>(1)
          )
        {
          // feature map size mismatch.
          return 8;
        }
      }
      else if (layers[i].type == NNLayer::maxPool)
      {
        layers[i].inMapSize = layers[i - 1].outMapSize;
        layers[i].outMapSize = layers[i].inMapSize / layers[i].filterSize;
        layers[i].numOutMaps = layers[i].numInMaps = layers[i - 1].numOutMaps;

        if (
          layers[i].inMapSize.at<int>(0) % layers[i].filterSize.at<int>(0) != 0 ||
          layers[i].inMapSize.at<int>(1) % layers[i].filterSize.at<int>(1) != 0
          )
          // inMapSize is not divisible by filterSize.
          return 13;
      }
    }
    else if (layers[i - 1].type == NNLayer::maxPool)
    {
      if (layers[i].type == NNLayer::perceptron || layers[i].type == NNLayer::linearPerceptron || layers[i].type == NNLayer::softMax)
      {
        if (layers[i - 1].outMapSize.at<int>(0) * layers[i - 1].outMapSize.at<int>(1) * layers[i - 1].numOutMaps != layers[i].inSize)
        {
          // Size mismatch.
          return 9;
        }
      }
      else if (layers[i].type == NNLayer::convolution)
      {
        if (layers[i - 1].numOutMaps != layers[i].numInMaps)
        {
          // number of feature maps mismatch.
          return 10;
        }
        else if (
          layers[i - 1].outMapSize.at<int>(0) != layers[i].inMapSize.at<int>(0) || 
          layers[i - 1].outMapSize.at<int>(1) != layers[i].inMapSize.at<int>(1)
          )
        {
          // feature map size mismatch.
          return 11;
        }
      }
      else if (layers[i].type == NNLayer::maxPool)
      {
        // successive maxPool layers is forbidden.
        return 12;
      }
    }
  }

  cv::RNG rng(time(NULL));
  for (int i = 0; i < numLayers(); ++i)
  {
    // initialize weights and biases.

    if (layers[i].type == NNLayer::perceptron || layers[i].type == NNLayer::linearPerceptron)
    {
#if 1 // Nguyen-Widrow algorithm
      double beta = (0.7 * pow(layers[i].outSize, 1.0 / layers[i].inSize));
      cv::Mat uBound = (cv::Mat_<real>(1, 1) <<1.0);
      cv::Mat lBound = (cv::Mat_<real>(1, 1) << -1.0);
      rng.fill(layers[i].weight, cv::RNG::UNIFORM, lBound, uBound);
      double norm = cv::norm(layers[i].weight);
      layers[i].weight *= (beta / norm);

      uBound.at<real>(0, 0) = (real)beta;
      lBound.at<real>(0, 0) = -(real)beta;
      rng.fill(layers[i].bias, cv::RNG::UNIFORM, lBound, uBound);
#else // uniform  distribution in [-2.4 / fan_in, 2.4 / fan_in]
      const real ub =  real(2.4) / layers[i].inSize;
      cv::Mat uBound = (cv::Mat_<real>(1, 1) << ub);
      cv::Mat lBound = (cv::Mat_<real>(1, 1) << -ub);
      //cv::Mat uBound = (cv::Mat_<real>(1, 1) <<  0.05);
      //cv::Mat lBound = (cv::Mat_<real>(1, 1) <<  -0.05);
      rng.fill(layers[i].weight, cv::RNG::UNIFORM, lBound, uBound);
      layers[i].bias = 0;
#endif
    }
    else if (layers[i].type == NNLayer::convolution)
    {
      // weight is <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.

      // uniform  distribution in [-2.4 / fan_in, 2.4 / fan_in]
      int fan_in = layers[i].filterSize.at<int>(0) * layers[i].filterSize.at<int>(1);
      const real ub =  real(2.4) / fan_in;
      cv::Mat uBound = (cv::Mat_<real>(1, 1) << ub);
      cv::Mat lBound = (cv::Mat_<real>(1, 1) << -ub);
      rng.fill(layers[i].weight, cv::RNG::UNIFORM, lBound, uBound);

      // bias is 1x<numOutMaps> matrix.
      layers[i].bias = 0;
    }
  }

  return 0;
}

/*
int NeuralNet::train_backprop(
  const cv::Mat &inputs,
  const cv::Mat &outputs,
  const cv::Mat &sampleWeights,
  const real learningRate,
  real &E)
{
  if (inputs.cols != layers[0].inSize)
  {
    // input size mismatch.
    return 1;
  }
  else if (outputs.cols != layers[numLayers() - 1].outSize)
  {
    // output size mismatch.
    return 2;
  }
  else if (inputs.rows != outputs.rows)
  {
    // number of samples mismatch.
    return 3;
  }

  // forward pass
  const cv::Mat *pX = &inputs;
  for (int i = 0; i < numLayers(); ++i)
    layers[i].forwardPropagate(pX);

  // backward pass
  
  //const int numSamples = inputs.rows;
  const int numOutputs = layers[numLayers() - 1].outSize;
  
  // set first grad
  cv::Mat firstGrad = layers[numLayers() - 1].grad; // just an alias
  //firstGrad = layers[numLayers() - 1].activation - outputs;
  layers[numLayers() - 1].activation.copyTo(firstGrad);
  firstGrad -= outputs;

  // calculate error

  // element-wise multiplication
  //cv::Mat sq = firstGrad.mul(firstGrad);
  cv::Mat sq; // (firstGrad.size(), firstGrad.type());
  elemMul(firstGrad, firstGrad, sq);

  cv::reduce(sq, sq, 1, CV_REDUCE_SUM);  // sum up each row
  sq = sq.mul(sampleWeights);
  cv::reduce(sq, sq, 0, CV_REDUCE_SUM);  // sum up the only column
  E = *(real *)sq.data;

  const int numSamples = inputs.rows;
  for (int i = 0; i < numSamples; ++i)
    firstGrad.row(i) *= ((real *)sampleWeights.data)[i];

  for (int i = numLayers() - 1; i >= 0; --i)
  {
    NNLayer &layer = layers[i];

    const cv::Mat *pPrecActivation = (i == 0)? (&inputs) : &(layers[i - 1].activation);

    // grad <- element-wise multiplication of grad and df
    //cv::multiply(layer.grad, layer.df, layer.grad);
    elemMul(layer.df, layer.grad);

    // dEdw <- [activation of preceding layer]^T * grad
    cv::Mat dEdw;
    matMul(*pPrecActivation, layer.grad, 1, dEdw, CV_GEMM_A_T);

    //std::cout << "dEdw : " << dEdw << std::endl;

    if (i > 0)
    {
      // [grad of prec layer] <- grad * weight^T
      //cv::gemm(layer.grad, layer.weight, 1, 0, 0, layers[i - 1].grad, CV_GEMM_B_T);
      matMul(layer.grad, layer.weight, 1, layers[i - 1].grad, CV_GEMM_B_T);
    }

    // update weights.
    layer.weight -= (dEdw * learningRate);

    // update biases.
    cv::Mat dEdb;
    cv::reduce(layer.grad, dEdb, 0, CV_REDUCE_SUM);  // sum up each column
    layer.bias -= (dEdb * learningRate);
  }

  return 0;
}
*/

static void UpdateBiasesRprop(const cv::Mat &grad, const updateParam &param, cv::Mat &db, cv::Mat &dbSign, cv::Mat &bias)
{
  cv::Mat dEdb;
  if (grad.dims == 2)
    cv::reduce(grad, dEdb, 0, CV_REDUCE_SUM);  // sum up each column
  else
  {
    // convolution layer.
    // grad is <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
    CV_Assert(grad.dims == 4);
    const int numSamples = grad.size[0];
    const int numOutMaps = grad.size[1];
    cv::Mat m1(numSamples, (int)grad.total() / numSamples, CV_REAL, grad.data);
    cv::Mat m2;
    cv::reduce(m1, m2, 0, CV_REDUCE_SUM);  // sum up each column (samples)
    cv::Mat m3(numOutMaps, (int)m2.total() / numOutMaps, CV_REAL, m2.data);
    cv::reduce(m3, dEdb, 1, CV_REDUCE_SUM);  // sum up each row (maps)
  }

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
  for (int j = 0; j < (int)dEdb.total(); ++j)
  {
    int sign = -CV_SIGN(((real *)dEdb.data)[j]);
    int ss = sign * ((int *)dbSign.data)[j];
    if (ss > 0)
    {
      real dval = ((real *)db.data)[j] * param.dw_plus;
      dval = std::min(dval, param.dw_max);
      ((real *)db.data)[j] = dval;
      ((real *)bias.data)[j] += dval * sign;
    }
    else if (ss < 0)
    {
      real dval = ((real *)db.data)[j] * param.dw_minus;
      dval = std::max(dval, param.dw_min);
      ((int *)dbSign.data)[j] = 0;
      ((real *)db.data)[j] = dval;
      ((real *)bias.data)[j] += dval * sign;
    }
    else
    {
      ((int *)dbSign.data)[j] = (char)sign;
      ((real *)bias.data)[j] += ((real *)db.data)[j] * sign;
    }
  }
}

int NeuralNet::train_sub(
  const cv::Mat &inputs,
  const cv::Mat &outputs,
  const cv::Mat &sampleWeights,
  const int firstLayerToTrain,
  const updateParam &update_param,
  cv::Mat &diffFromTarget,
  real &E    // in, out : the first time this func is called, E should be 0.
  )
{
  //Log("Head of train_sub().\n");

  if (
    layers[0].type == NNLayer::perceptron ||
    layers[0].type == NNLayer::linearPerceptron ||
    layers[0].type == NNLayer::softMax)
  {
    if (inputs.cols != layers[0].inSize)
      // input size mismatch.
      return 1;
  }
  else if (
    layers[numLayers() - 1].type != NNLayer::perceptron &&
    layers[numLayers() - 1].type != NNLayer::linearPerceptron &&
    layers[numLayers() - 1].type != NNLayer::softMax)
  {
    // The last layer should be either percertron or softMax.
    return 4;
  }
  else if (outputs.cols != layers[numLayers() - 1].outSize)
  {
    // output size mismatch.
    return 5;
  }
  else if (inputs.rows != outputs.rows)
  {
    // number of samples mismatch.
    return 6;
  }

  // forward pass
  //Log("forward propagation\n");

  const cv::Mat *pX = &inputs;
  if (E == 0) // the first time !!!minibatchにも対応が必要!!!
  {
    // forward propagate on layer 0 ... firstLayerToTrain - 1
    for (int i = 0; i < firstLayerToTrain; ++i)
    {
      //printf("layer %d (type %d)\n", i, layers[i].type);
      layers[i].forwardPropagate(pX, false, false);
    }
  }

  // forward propagate on layer firstLayerToTrain ... maxLayer
  if (firstLayerToTrain > 0)
    pX = &(layers[firstLayerToTrain - 1].activation);

  for (int i = firstLayerToTrain; i < numLayers(); ++i)
  {
    //printf("layer %d (type %d)\n", i, layers[i].type);
    layers[i].forwardPropagate(pX, true, true);  
  }

  // prepare for backward pass
  
  //Log("calculate the first gradient\n");

  //const int numSamples = inputs.rows;
  const int numOutputs = layers[numLayers() - 1].outSize;
  
  // set first grad
  cv::Mat firstGrad = layers[numLayers() - 1].grad; // just an alias
  //-----------------------------------------------------------
  // firstGrad = layers[numLayers() - 1].activation - outputs;
  // としてはいけない。firstGrad.data がつけ変わってしまう。
  //-----------------------------------------------------------
  //std::cout << std::endl << "layers[numLayers() - 1].activation : " << layers[numLayers() - 1].activation << std::endl;
  layers[numLayers() - 1].activation.copyTo(firstGrad);
  firstGrad -= outputs;
  firstGrad.copyTo(diffFromTarget);

  //std::cout << "activ_0 : " << layers[numLayers() - 1].activation.row(0) << std::endl;
  //std::cout << "outputs_0 : " << outputs.row(0) << std::endl;

  // calculate error
  //Log("calculate the error\n");

  const int numSamples = inputs.rows;

  if (layers[numLayers() - 1].type == NNLayer::softMax)
  {
    // Calculate the cross-entropy error function:
    // E = - Sigma_k {t_k * log(activation_k)}
    cv::Mat softMaxActivation;
    layers[numLayers() - 1].activation.copyTo(softMaxActivation);

    //std::cout << "softmax activation : " << std::endl;
    //printMat(softMaxActivation);

    cv::log(softMaxActivation, softMaxActivation);

    //std::cout << "log: " << std::endl;
    //printMat(softMaxActivation);

    //std::cout << "outputs: " << std::endl;
    //printMat(outputs);

    E = 0;
    for (int s = 0; s < numSamples; ++s)
    {
      E += -(real)outputs.row(s).dot(softMaxActivation.row(s)) * sampleWeights.at<real>(s);
    }
  }
  else
  {
    // Calculate the sum-square error function.

    cv::Mat sq;
    elemMul(firstGrad, firstGrad, sq); // element-wise multiplication

    //std::cout << "firstGrad_0 : " << firstGrad.row(0) << std::endl;
    //std::cout << "sq_0 : " << sq.row(0) << std::endl;

    cv::reduce(sq, sq, 1, CV_REDUCE_SUM);  // sum up each row
    sq = sq.mul(sampleWeights);
    cv::reduce(sq, sq, 0, CV_REDUCE_SUM);  // sum up the only column
    E = *(real *)sq.data;
  }

  // Apply sampleWeights to the firstGrad.
  for (int i = 0; i < numSamples; ++i)
  {
    cv::Mat tmp(1, (int)firstGrad.total() / numSamples, firstGrad.type(), firstGrad.data + firstGrad.step[0] * i);
    tmp *= ((real *)sampleWeights.data)[i];
  }

  // backward pass

  //Log("back propagation\n");

  for (int i = numLayers() - 1; i >= firstLayerToTrain; --i)
  {
    NNLayer &layer = layers[i];

    //printf("layer %d (type %d)\n", i, layer.type);

    const cv::Mat *pPrecActivation = (i == 0)? (&inputs) : &(layers[i - 1].activation);

    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::convolution)
    {
      //Log("grad <- element-wise multiplication of grad and df\n");

      // grad <- element-wise multiplication of grad and df
      //
      // In case of convolution layer, grad, df, activation are
      // <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
      //std::cout << "layer.df : " << layer.df << std::endl;
      //std::cout << "layer.grad 1 : " << layer.grad << std::endl;
      elemMul(layer.df, layer.grad);
      //std::cout << "layer.grad 2 : " << layer.grad << std::endl;
    }

    //if (layer.type != NNLayer::softMax)
    //  Log("dEdw <- [activation of preceding layer]^T * grad\n");

    cv::Mat dEdw;
    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
    {
      cv::Mat tmp;
      if (pPrecActivation->dims == 4)
      {
        // preceding layer is a convolution layer.
        tmp = cv::Mat(numSamples, (int)pPrecActivation->total() / numSamples, CV_REAL, pPrecActivation->data);
        pPrecActivation = &tmp;
      }

      // dEdw <- [activation of preceding layer]^T * grad
      //std::cout << std::endl << "*pPrecActivation : " << *pPrecActivation << std::endl;
      //std::cout << std::endl << "layer.grad : " << layer.grad << std::endl;
      matMul(*pPrecActivation, layer.grad, 1, dEdw, CV_GEMM_A_T);
      //std::cout << std::endl << "dEdw : " << dEdw << std::endl;
    }
    else if (layer.type == NNLayer::convolution)
    {
      // *pPrecActivation is a <num samples>x<numInMaps>x<in_H>x<in_W> matrix.

      // dEdw is a <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
      cv::Mat matSize = (cv::Mat_<int>(1, 4) << layer.numOutMaps, layer.numInMaps, layer.filterSize.at<int>(0), layer.filterSize.at<int>(1));
      dEdw.create(4, (int *)matSize.data, CV_REAL);
      dEdw = 0;

      // !!! 遅い
#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
      for (int s = 0; s < numSamples; ++s)
      {
        for (int outMap = 0; outMap < layer.numOutMaps; ++outMap)
        {
          cv::Mat currentGrad(layer.outMapSize.at<int>(0), layer.outMapSize.at<int>(1), CV_REAL, layer.grad.data + layer.grad.step[0] * s + layer.grad.step[1] * outMap);

          for (int inMap = 0; inMap < layer.numInMaps; ++inMap)
          {
            cv::Mat currentPrecActiv(layer.inMapSize.at<int>(0), layer.inMapSize.at<int>(1), CV_REAL, pPrecActivation->data + pPrecActivation->step[0] * s + pPrecActivation->step[1] * inMap);
            cv::Mat currentdEdw(layer.filterSize.at<int>(0), layer.filterSize.at<int>(1), CV_REAL, dEdw.data + dEdw.step[0] * outMap + dEdw.step[1] * inMap);

            for (int y = 0; y < layer.filterSize.at<int>(0); ++y)
            {
              for (int x = 0; x < layer.filterSize.at<int>(1); ++x)
              {
                currentdEdw.at<real>(y, x) += (real)currentPrecActiv(cv::Rect(x, y, currentGrad.cols, currentGrad.rows)).dot(currentGrad);
              }
            }
          }
        }
      }
    }

    //std::cout << "dEdw : " << dEdw << std::endl;

    if (i > 0)
    {
      NNLayer &precLayer = layers[i - 1];

      //Log("[grad of prec layer] <- grad * weight^T\n");

      if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
      {
        // [grad of prec layer] <- grad * weight^T
        matMul(layer.grad, layer.weight, 1, precLayer.grad, CV_GEMM_B_T);

        if (precLayer.type == NNLayer::convolution || precLayer.type == NNLayer::maxPool)
        {
          // reshape precLayer.grad to be <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
          cv::Mat matSize = (cv::Mat_<int>(1, 4) << numSamples, precLayer.numOutMaps, precLayer.outMapSize.at<int>(0), precLayer.outMapSize.at<int>(1));
          precLayer.grad = reshape(precLayer.grad, 4, (int *)(matSize.data));
        }
      }
      else if (layer.type == NNLayer::softMax)
      {
        // [grad of prec layer] <- grad
        layer.grad.copyTo(precLayer.grad);
        //std::cout << std::endl << "softMax: layer.grad (after update) : " << layer.grad << std::endl;

        if (precLayer.type == NNLayer::convolution || precLayer.type == NNLayer::maxPool)
        {
          // reshape precLayer.grad to be <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
          cv::Mat matSize = (cv::Mat_<int>(1, 4) << numSamples, precLayer.numOutMaps, precLayer.outMapSize.at<int>(0), precLayer.outMapSize.at<int>(1));
          precLayer.grad = reshape(precLayer.grad, 4, (int *)(matSize.data));
        }
      }
      else if (layer.type == NNLayer::convolution)
      {
        const int v_margin = (layer.filterSize.at<int>(0) - 1) / 2;
        const int h_margin = (layer.filterSize.at<int>(1) - 1) / 2;
        precLayer.grad = 0;

        // !!! 遅い
#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
        for (int s = 0; s < numSamples; ++s)
        {
          for (int inMap = 0; inMap < layer.numInMaps; ++inMap)
          {
            cv::Mat currentPrecGrad(layer.inMapSize.at<int>(0), layer.inMapSize.at<int>(1), CV_REAL, precLayer.grad.data + precLayer.grad.step[0] * s + precLayer.grad.step[1] * inMap);
            //currentPrecGrad = 0;

            for (int outMap = 0; outMap < layer.numOutMaps; ++outMap)
            {
              cv::Mat currentGrad(layer.outMapSize.at<int>(0), layer.outMapSize.at<int>(1), CV_REAL, layer.grad.data + layer.grad.step[0] * s + layer.grad.step[1] * outMap);
              cv::Mat currentFilter(layer.filterSize.at<int>(0), layer.filterSize.at<int>(1), CV_REAL, layer.weight.data + layer.weight.step[0] * outMap + layer.weight.step[1] * inMap);

              cv::Mat tmpGrad;
              copyMakeBorder(currentGrad, tmpGrad, v_margin, v_margin, h_margin, h_margin, cv::BORDER_CONSTANT, cv::Scalar(0));

              cv::Mat tmpFilter;
              cv::flip(currentFilter, tmpFilter, -1); // 両方の軸で反転
              cv::Ptr<cv::FilterEngine> pEngine = createLinearFilter(CV_REAL, CV_REAL, tmpFilter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
              cv::Mat tmpPrecGrad(currentPrecGrad.size(), currentPrecGrad.type());
              pEngine->apply(tmpGrad, tmpPrecGrad);
              currentPrecGrad += tmpPrecGrad;
            }
          }
        }
      }
      else if (layer.type == NNLayer::maxPool)
      {
        const int out_H = layer.outMapSize.at<int>(0);
        const int out_W = layer.outMapSize.at<int>(1);
        const int filter_H = layer.filterSize.at<int>(0);
        const int filter_W = layer.filterSize.at<int>(1);

        precLayer.grad = 0;

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
        for (int s = 0; s < numSamples; ++s)
        {
          for (int map = 0; map < layer.numInMaps; ++map)
          {
            cv::Mat currentPrecGrad(layer.inMapSize.at<int>(0), layer.inMapSize.at<int>(1), CV_REAL, precLayer.grad.data + precLayer.grad.step[0] * s + precLayer.grad.step[1] * map);
            cv::Mat currentGrad(layer.outMapSize.at<int>(0), layer.outMapSize.at<int>(1), CV_REAL, layer.grad.data + layer.grad.step[0] * s + layer.grad.step[1] * map);
            cv::Mat currentDf(layer.outMapSize.at<int>(0), layer.outMapSize.at<int>(1), CV_32S, layer.df.data + layer.df.step[0] * s + layer.df.step[1] * map);

            for (int y = 0; y < out_H; ++y)
            {
              for (int x = 0; x < out_W; ++x)
              {
                const int pos = currentDf.at<int>(y, x);
                const int posX = pos %  filter_W;
                const int posY = pos /  filter_W;
                currentPrecGrad.at<real>(y * filter_H + posY, x * filter_W + posX) = currentGrad.at<real>(y, x);
              }
            }
          }
        }
      }
    }

    if (
      layer.type == NNLayer::perceptron ||
      layer.type == NNLayer::linearPerceptron ||
      layer.type == NNLayer::convolution)
    {
      //Log("update weights and biases\n");

      if (update_param.type == updateParam::rprop)
      {
        // update weights.
        layer.UpdateWeightsRprop(dEdw, update_param);
      
        //std::cout << "layer.weight : " << std::endl;
        //printMat(layer.weight);

        // update biases.
        UpdateBiasesRprop(layer.grad, update_param, layer.db, layer.dbSign, layer.bias);
      }
      else if (update_param.type == updateParam::bprop)
      {
        // update weights.

        //std::cout << std::endl << "dEdw : " << dEdw << std::endl;
        //std::cout << std::endl << "layer.last_dW : " << layer.last_dW << std::endl;
        layer.last_dW = (layer.last_dW * update_param.momentum) - (dEdw * (update_param.learningRate * (1 - update_param.momentum)));
        //std::cout << std::endl << "layer.last_dW (after update) : " << layer.last_dW << std::endl;
        
        //std::cout << std::endl << "layer.weight : " << layer.weight << std::endl;
        layer.weight += layer.last_dW;
        //std::cout << std::endl << "layer.weight (after update) : " << layer.weight << std::endl;

        // update biases.
        cv::Mat dEdb;
        cv::reduce(layer.grad, dEdb, 0, CV_REDUCE_SUM);  // sum up each column
        layer.last_db = (layer.last_db * update_param.momentum) - (dEdb * (update_param.learningRate * (1 - update_param.momentum)));
        layer.bias += layer.last_db;
      }
    }

    if (layer.type == NNLayer::linearPerceptron && layer.maxWeightNorm != 0)
    {
      double weightNorm = cv::norm(layer.weight);
      if (layer.maxWeightNorm < weightNorm)
      {
        double coef = layer.maxWeightNorm / weightNorm;
        layer.weight *= coef;
        layer.bias *= coef;
      }
    }
  }

  //Log("End of train_sub().\n");

  return 0;
}

static void ChooseSamples(
  const cv::Mat &inMat,
  cv::Mat &outMat,
  const int sampleHead,
  const int minibatchSize,
  const std::vector<unsigned int> &permutation,
  const bool cyclic = true)
{
  const int numSamples = inMat.size[0];
  int numOutRows;

  if (cyclic)
  {
    numOutRows = minibatchSize;
    outMat.create(minibatchSize, (int)inMat.total() / numSamples, inMat.type());
  }
  else
  {
    numOutRows = (sampleHead + minibatchSize < numSamples)? minibatchSize : (numSamples - sampleHead);
    outMat.create(numOutRows, (int)inMat.total() / numSamples, inMat.type());
  }

  for (int i = 0; i < numOutRows; ++i)
  {
    unsigned int index = permutation[(sampleHead + i) % numSamples];
    inMat.row(index).copyTo(outMat.row(i));
  }
}

static void AdjustSampleWeights(cv::Mat &sampleWeights, const cv::Mat &firstGrad)
{
  CV_Assert(sampleWeights.rows == firstGrad.rows);

  cv::Mat rowMax, rowMin, coef;
  cv::reduce(firstGrad, rowMax, 1, CV_REDUCE_MAX);
  cv::reduce(firstGrad, rowMin, 1, CV_REDUCE_MIN);

#if 0
  cv::log(rowMin - rowMax + 2, coef);
  coef = 1 - (coef / log(2.0));
#else
  const real a = 10;
  cv::exp(-a * (rowMin - rowMax + 1), coef);
#endif

  //std::cout << "coef: " << coef.rowRange(0, 100) << std::endl;

  elemMul(coef, sampleWeights);
}

int NeuralNet::train(
  const cv::Mat &inputs,         // [num samples] x [input vector size] matrix
  void (*TransformSamples)(cv::Mat &inputs, void *transformSamplesInfo), // callback function for transforming input data
  void *transformSamplesInfo,    // extra info passed to TransformSamples()
  const cv::Mat &outputs,        // [num samples] x [output vector size] matrix
  const cv::Mat &sampleWeights,  // column vector of size [num samples]
  const int firstLayerToTrain,
  const updateParam &update_param_,
  const int maxIter,
  const int evaluateEvery,
  void (*funcToEvaluateEvery)(NeuralNet &nn), // callback function to be called every 'evaluateEvery' epochs
  const bool initializeLearningState,
  const int minibatchSize,
  real &E                        // out : error.
  )
{
  // Copy so that values can be changed.
  updateParam update_param = update_param_;

  // normalize sampleWeights
  cv::Mat sumMat;
  cv::reduce(sampleWeights, sumMat, 0, CV_REDUCE_SUM);  // sum up the only column
  sampleWeights /= sumMat.at<real>(0, 0);

  const int numLayers = NeuralNet::numLayers();
  const int numSamples = inputs.rows;
  const int batchSize = (minibatchSize == 0)? numSamples : minibatchSize;
  for (int i = 0; i < numLayers; ++i)
  {
    if (
      layers[i].type == NNLayer::perceptron ||
      layers[i].type == NNLayer::linearPerceptron)
    {
      layers[i].activation.create(batchSize, layers[i].outSize, CV_REAL);
      if (layers[i].activation.data == 0)
        return 1;
      if (layers[i].type == NNLayer::perceptron)
      {
        layers[i].df.create(batchSize, layers[i].outSize, CV_REAL);
        if (layers[i].df.data == 0)
          return 2;
      }
      layers[i].grad.create(batchSize, layers[i].outSize, CV_REAL);
      if (layers[i].grad.data == 0)
        return 3;

      if (update_param.type == updateParam::rprop)
      {
        bool doInitialize = false;
        if (layers[i].dw.data == NULL || initializeLearningState)
          doInitialize = true;

        layers[i].dw.create(layers[i].inSize, layers[i].outSize, CV_REAL);
        layers[i].dwSign.create(layers[i].inSize, layers[i].outSize, CV_32S);
        layers[i].db.create(1, layers[i].outSize, CV_REAL);
        layers[i].dbSign.create(1, layers[i].outSize, CV_32S);

        if (doInitialize)
        {
          layers[i].dw = update_param.dw0;
          layers[i].dwSign = 0;
          layers[i].db = update_param.dw0;
          layers[i].dbSign = 0;
        }
      }
      else if (update_param.type == updateParam::bprop)
      {
        bool doInitialize = false;
        if (layers[i].last_dW.data == NULL || initializeLearningState)
          doInitialize = true;

        layers[i].last_dW.create(layers[i].inSize, layers[i].outSize, CV_REAL);
        layers[i].last_db.create(1, layers[i].outSize, CV_REAL);
        if (doInitialize)
        {
          layers[i].last_dW = 0;
          layers[i].last_db = 0;
        }
      }
    }
    else if (layers[i].type == NNLayer::softMax)
    {
      layers[i].activation.create(batchSize, layers[i].outSize, CV_REAL);
      if (layers[i].activation.data == 0)
        return 4;
      layers[i].grad.create(batchSize, layers[i].outSize, CV_REAL);
      if (layers[i].grad.data == 0)
        return 5;
    }
    else if (layers[i].type == NNLayer::convolution)
    {
      if (i == 0)
      {
        if (layers[i].inMapSize.at<int>(0) * layers[i].inMapSize.at<int>(1) != inputs.cols)
          return 6;

        if (layers[i].numInMaps != 1)
          return 7;
      }

      // activation, df, grad are <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
      cv::Mat matSize = (cv::Mat_<int>(1, 4) << batchSize, layers[i].numOutMaps, layers[i].outMapSize.at<int>(0), layers[i].outMapSize.at<int>(1));
      layers[i].activation.create(4, (int *)matSize.data, CV_REAL);
      if (layers[i].activation.data == 0)
        return 8;

      layers[i].df.create(4, (int *)matSize.data, CV_REAL);
      if (layers[i].df.data == 0)
        return 9;

      layers[i].grad.create(4, (int *)matSize.data, CV_REAL);
      if (layers[i].grad.data == 0)
        return 10;

      cv::Mat weightSize = (cv::Mat_<int>(1, 4) << layers[i].numOutMaps, layers[i].numInMaps, layers[i].filterSize.at<int>(0), layers[i].filterSize.at<int>(1));
      if (update_param.type == updateParam::rprop)
      {
        bool doInitialize = false;
        if (layers[i].dw.data == NULL || initializeLearningState)
          doInitialize = true;

        // dw and dwSign are <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
        layers[i].dw.create(4, (int *)(weightSize.data), CV_REAL);
        layers[i].dwSign.create(4, (int *)(weightSize.data), CV_32S);

        // db and dbSign are 1x<numOutMaps> matrix.
        layers[i].db.create(1, layers[i].numOutMaps, CV_REAL);
        layers[i].dbSign.create(1, layers[i].numOutMaps, CV_32S);

        if (doInitialize)
        {
          layers[i].dw = update_param.dw0;
          layers[i].dwSign = 0;
          layers[i].db = update_param.dw0;
          layers[i].dbSign = 0;
        }
      }
      else if (update_param.type == updateParam::bprop)
      {
        bool doInitialize = false;
        if (layers[i].last_dW.data == NULL || initializeLearningState)
          doInitialize = true;

        layers[i].last_dW.create(4, (int *)(weightSize.data), CV_REAL);
        layers[i].last_db.create(1, layers[i].numOutMaps, CV_REAL);

        if (doInitialize)
        {
          layers[i].last_dW = 0;
          layers[i].last_db = 0;
        }
      }
    }
    else if (layers[i].type == NNLayer::maxPool)
    {
      if (i == 0)
      {
        layers[i].inMapSize.at<int>(1) = inputs.cols;
        if (inputs.cols % layers[i].filterSize.at<int>(1) != 0)
          return 11;
        layers[i].outMapSize.at<int>(1) = inputs.cols / layers[i].filterSize.at<int>(1);
      }

      // activation, df and grad are <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
      cv::Mat matSize = (cv::Mat_<int>(1, 4) << batchSize, layers[i].numOutMaps, layers[i].outMapSize.at<int>(0), layers[i].outMapSize.at<int>(1));
      layers[i].activation.create(4, (int *)matSize.data, CV_REAL);
      // df has a different meaning for maxPool.
      // maxPool has no activation function. It outputs just the max of input values
      // so that df is always 1.
      // Instead, we use df to memorize the position of the input value chosen
      // by the max operation.
      // In the backward propagation, grad of the output maps will be just copied
      // to the corresponding chosen input; grad of other inputs shall be 0.
      layers[i].df.create(4, (int *)matSize.data, CV_32S);
      layers[i].grad.create(4, (int *)matSize.data, CV_REAL);
      if (layers[i].activation.data == 0 || layers[i].df.data == 0 || layers[i].grad.data == 0)
        return 12;
    }
  }

  std::vector<unsigned int> permutation(numSamples);
	cv::RNG rng(time(NULL));
  GetRandomPermutation(permutation, rng);

  E = 0;
  cv::Mat inputs1;
  cv::Mat sampleWeights1;
  cv::Mat diffFromTarget(numSamples, 1, CV_REAL);
  cv::Mat outputs1 = outputs;
  int sampleHead = 0; // the index of the first sample in the current minibatch
  for (int iter = 0; iter < maxIter; ++iter)
  {
    if (minibatchSize != 0)
    {
      ChooseSamples(inputs, inputs1, sampleHead, minibatchSize, permutation);
      ChooseSamples(outputs, outputs1, sampleHead, minibatchSize, permutation);
      ChooseSamples(sampleWeights, sampleWeights1, sampleHead, minibatchSize, permutation);
      sampleHead = (sampleHead + minibatchSize) % numSamples;
    }
    else
    {
      inputs1 = inputs.clone();
      sampleWeights1 = sampleWeights;
      
      // predict に失敗したサンプルについて sampleWeights を重くする。
      //sampleWeights1 = sampleWeights.clone();
      //if (iter != 0)
      //  AdjustSampleWeights(sampleWeights1, diffFromTarget);
    }

    if (TransformSamples != NULL && iter != 0)
      (*TransformSamples)(inputs1, transformSamplesInfo);

    int ret = train_sub(inputs1, outputs1, sampleWeights1, firstLayerToTrain, update_param, diffFromTarget, E);
    if (ret != 0)
      return (ret + 100);

    if (update_param.type == updateParam::bprop)
    {
      printf("%d. E = %f (learning rate=%f, momentum=%f)\n", iter, E, update_param.learningRate, update_param.momentum);
      Log("%d. E = %f (learning rate=%f, momentum=%f)\n", iter, E, update_param.learningRate, update_param.momentum);

      update_param.learningRate *= update_param.learningRateDecay;
      if (update_param.learningRate < update_param.finalLearningRate)
        update_param.learningRate = update_param.finalLearningRate;

      update_param.momentum += update_param.momentumDelta;
      if (update_param.momentum > update_param.finalMomentum)
        update_param.momentum = update_param.finalMomentum;
    }
    else if (update_param.type == updateParam::rprop)
    {
      printf("%d. E = %f\n", iter, E);
      Log("%d. E = %f\n", iter, E);
    }

    if (evaluateEvery > 0 && funcToEvaluateEvery != NULL && ((iter + 1) % evaluateEvery) == 0)
      (*funcToEvaluateEvery)(*this);

    if (_access("stopLoop", 0) != -1)
    {
      DeleteFileA("stopLoop");
      break;
    }

  }

  return 0;
}

int NeuralNet::autoencode_one_layer(
  const int layerNum,
  const cv::Mat &inputs0,         // [num samples] x [input vector size] matrix
  const cv::Mat &sampleWeights0,  // column vector of size [num samples]
  const updateParam &update_param_,
  const int maxIter,
  const int minibatchSize,
  real &E                         // out : error.
  )
{
  const bool initializeLearningState = true;

  // Copy so that values can be changed.
  updateParam update_param = update_param_;

  NNLayer &layer = layers[layerNum];
	cv::RNG rng(time(NULL));

  const int numSamples = inputs0.size[0];
  const int batchSize = (minibatchSize == 0)? numSamples : minibatchSize;

  const int in_H = (layer.type == NNLayer::convolution)? layer.inMapSize.at<int>(0) : 0;
  const int in_W = (layer.type == NNLayer::convolution)? layer.inMapSize.at<int>(1) : 0;
  const int out_H = (layer.type == NNLayer::convolution)? layer.outMapSize.at<int>(0) : 0;
  const int out_W = (layer.type == NNLayer::convolution)? layer.outMapSize.at<int>(1) : 0;
  const int filter_H = (layer.type == NNLayer::convolution)? layer.filterSize.at<int>(0) : 0;
  const int filter_W = (layer.type == NNLayer::convolution)? layer.filterSize.at<int>(1) : 0;

  //--------------------------------------------------------------------
  // impriments autoencoder.
  //
  // forward propagation
  //   input0           --(*W, +layer.bias)--> layer.activation, layer.df
  //   layer.activation --(*W^T, +bias2   )--> y2, activation2, df2
  //
  // backword propagation
  //   activation2 - inputs0 -> grad2
  //   ||grad2||_2           -> Error 
  //   [grad2_ij*df2_ij]     -> grad2
  //   grad2^T * layer.activation -> dEdw2
  //   ------
  //   grad2 * layer.weight  -> layer.grad
  //   [layer.grad_ij*layer.df_ij] -> layer.grad
  //   inputs^T * layer.grad -> dEdw 
  // update
  //   layer.W      by dEdw2 + dEdw
  //   bias2        by grad2
  //   layer.bias   by layer.grad
  //--------------------------------------------------------------------

  cv::Mat y2;
  cv::Mat bias2;
  cv::Mat activation2;
  cv::Mat df2;
  cv::Mat db2;
  cv::Mat dbSign2;

  if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
  {
    layer.activation.create(batchSize, layer.outSize, CV_REAL);
    if (layer.activation.data == 0)
      return 1;
    if (layer.type == NNLayer::perceptron)
    {
      layer.df.create(batchSize, layer.outSize, CV_REAL);
      if (layer.df.data == 0)
        return 2;
    }
    layer.grad.create(batchSize, layer.outSize, CV_REAL);
    if (layer.grad.data == 0)
      return 3;

    if (update_param.type == updateParam::rprop)
    {
      bool doInitialize = false;
      if (layer.dw.data == NULL || initializeLearningState)
        doInitialize = true;

      layer.dw.create(layer.inSize, layer.outSize, CV_REAL);
      layer.dwSign.create(layer.inSize, layer.outSize, CV_32S);
      layer.db.create(1, layer.outSize, CV_REAL);
      layer.dbSign.create(1, layer.outSize, CV_32S);

      if (doInitialize)
      {
        layer.dw = update_param.dw0;
        layer.dwSign = 0;
        layer.db = update_param.dw0;
        layer.dbSign = 0;
      }
    }
    else if (update_param.type == updateParam::bprop)
    {
      bool doInitialize = false;
      if (layer.last_dW.data == NULL || initializeLearningState)
        doInitialize = true;

      layer.last_dW.create(layer.inSize, layer.outSize, CV_REAL);
      layer.last_db.create(1, layer.outSize, CV_REAL);

      if (doInitialize)
      {
        layer.last_dW = 0;
        layer.last_db = 0;
      }
    }

    y2 = cv::Mat(batchSize, layer.inSize, CV_REAL);
    bias2 = cv::Mat(1, layer.inSize, CV_REAL, cv::Scalar::all(0));
    activation2 = cv::Mat(batchSize, layer.inSize, CV_REAL);
    if (layer.type == NNLayer::perceptron)
      df2 = cv::Mat(batchSize, layer.inSize, CV_REAL);
    db2 = cv::Mat(1, layer.inSize, CV_REAL, cv::Scalar::all(update_param.dw0));
    dbSign2 = cv::Mat(1, layer.inSize, CV_32S, cv::Scalar::all(0));
  }
#if 0
  else if (layer.type == NNLayer::convolution)
  {
    // activation, df, grad are <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
    cv::Mat matSize = (cv::Mat_<int>(1, 4) << batchSize, layer.numOutMaps, out_H, out_W);
    layer.activation.create(4, (int *)matSize.data, CV_REAL);
    if (layer.activation.data == 0)
      return 5;

    layer.df.create(4, (int *)matSize.data, CV_REAL);
    if (layer.df.data == 0)
      return 6;

    layer.grad.create(4, (int *)matSize.data, CV_REAL);
    if (layer.grad.data == 0)
      return 7;

    // for rprop
    // dw and dwSign are <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
    cv::Mat weightSize = (cv::Mat_<int>(1, 4) << layer.numOutMaps, layer.numInMaps, filter_H, filter_W);
    layer.dw.create(4, (int *)(weightSize.data), CV_REAL);
    layer.dw = update_param.dw0;
    layer.dwSign.create(4, (int *)(weightSize.data), CV_32S);
    layer.dwSign = 0;

    // db and dbSign are 1x<numOutMaps> matrix.
    layer.db.create(1, layer.numOutMaps, CV_REAL);
    layer.db = update_param.dw0;
    layer.dbSign.create(1, layer.numOutMaps, CV_32S);
    layer.dbSign = 0;

    // y2 holds the estimated value of x.
    weightSize = (cv::Mat_<int>(1, 4) << batchSize, layer.numInMaps, in_H, in_W);
    y2 = cv::Mat(4, (int *)(weightSize.data), CV_REAL);
    bias2 = cv::Mat(1, layer.numInMaps, CV_REAL, cv::Scalar::all(0));
    activation2 = cv::Mat(4, (int *)(weightSize.data), CV_REAL);
    df2 = cv::Mat(4, (int *)(weightSize.data), CV_REAL);
    db2 = cv::Mat(1, layer.numInMaps, CV_REAL, cv::Scalar::all(update_param.dw0));
    dbSign2 = cv::Mat(1, layer.numInMaps, CV_32S, cv::Scalar::all(0));
  }
#endif
  else
  {
    printf("autoencode_one_layer: Only perceptron layers are supported.\n");
    Log("autoencode_one_layer: Only perceptron layers are supported.\n");
    return 8;
  }

  std::vector<unsigned int> permutation(numSamples);
  GetRandomPermutation(permutation, rng);

  cv::Mat inputs_sub = inputs0;
  cv::Mat sampleWeights_sub = sampleWeights0;
  int sampleHead = 0; // the index of the first sample in the current minibatch
  for (int iter = 0; iter < maxIter; ++iter)
  {
    if (minibatchSize != 0)
    {
      ChooseSamples(inputs0, inputs_sub, sampleHead, minibatchSize, permutation);
      ChooseSamples(sampleWeights0, sampleWeights_sub, sampleHead, minibatchSize, permutation);
      sampleHead = (sampleHead + minibatchSize) % numSamples;
    }

    cv::Mat dropped_inputs_sub = inputs_sub;
    if (layer.dropoutRatio != 0)
      dropped_inputs_sub = inputs_sub.clone();

    // forward path

    // set value of layer.df and layer.activation,
    // applying Dropout.
    //
    // inputs can be modified by Dropout.
    const cv::Mat *pX = &dropped_inputs_sub;
    layer.forwardPropagate(pX, true, true);

    // compute
    //   y2 = layer.activation * W^T + bias2,
    //   activation2 = f(y2),
    //   df2 = f'(y2) .

    // y2 <- layer.activation * W^T
    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
    {
      matMul(layer.activation, layer.weight, 1, y2, CV_GEMM_B_T);
    }
    else if (layer.type == NNLayer::convolution)
    {
      y2 = 0;
      const int v_margin = (layer.filterSize.at<int>(0) - 1) / 2;
      const int h_margin = (layer.filterSize.at<int>(1) - 1) / 2;

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
      for (int s = 0; s < batchSize; ++s)
      {
        for (int inMap = 0; inMap < layer.numInMaps; ++inMap)
        {
          cv::Mat y2_(in_H, in_W, CV_REAL, y2.data + y2.step[0] * s + y2.step[1] * inMap);

          for (int outMap = 0; outMap < layer.numOutMaps; ++outMap)
          {
            cv::Mat activ_(out_H, out_W, CV_REAL, layer.activation.data + layer.activation.step[0] * s + layer.activation.step[1] * outMap);
            cv::Mat filter_(filter_H, filter_W, CV_REAL, layer.weight.data + layer.weight.step[0] * outMap + layer.weight.step[1] * inMap);

            cv::Mat tmpActiv;
            copyMakeBorder(activ_, tmpActiv, v_margin, v_margin, h_margin, h_margin, cv::BORDER_CONSTANT, cv::Scalar(0));

            cv::Mat tmpFilter;
            cv::flip(filter_, tmpFilter, -1); // 両方の軸で反転
            cv::Ptr<cv::FilterEngine> pEngine = createLinearFilter(CV_REAL, CV_REAL, tmpFilter, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::Mat tmpY2(y2_.size(), y2_.type());
            pEngine->apply(tmpActiv, tmpY2);
            y2_ += tmpY2;
          }
        }
      }
    }

    // y2 += bias2
    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
    {
      for (int r = 0; r < batchSize; ++r)
        y2.row(r) += bias2;
    }
    else if (layer.type == NNLayer::convolution)
    {
#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
      for (int s = 0; s < batchSize; ++s)
      {
        for (int inMap = 0; inMap < layer.numInMaps; ++inMap)
        {
          cv::Mat y2_(in_H, in_W, CV_REAL, y2.data + y2.step[0] * s + y2.step[1] * inMap);
          y2_ += bias2.at<real>(inMap);
        }
      }
    }

    // activation2 = f(y2),
    // df2 = f'(y2) .
    //
    // TO DO: defined(HAVE_CPPAMP) && defined(REAL_IS_FLOAT) の場合
    // NNLayer::activateTanh() を参考に高速化する。
    cv::Mat exp_y = (real(4) / 3) * y2;
    cv::exp(exp_y, exp_y);
    // -----------------------------------
    // スカラーを行列で割ると、値の大きな要素の結果が #IND となることがある。
    // それよりも大きな値だと正しく 0 になる。
    // activation2 = 1 - 2 / (exp_y + 1);
    // 下のコードはその回避策。
    // ------------------------------------
#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < (int)exp_y.total(); ++i)
      *(real *)(activation2.data + i * sizeof(real)) = 1 - 2 / (*(real *)(exp_y.data + i * sizeof(real)) + 1);

    if (layer.type == NNLayer::perceptron)
      df2 = (real(1.7159) * 2 / 3) * (1 - activation2.mul(activation2));
    activation2 *= real(1.7159);

    // calculate error
    cv::Mat grad2;
    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
    {
      if (inputs_sub.dims == 4)
      {
        cv::Mat tmp(batchSize, (int)inputs_sub.total() / batchSize, CV_REAL, inputs_sub.data);
        grad2 = activation2 - tmp;
      }
      else
        grad2 = activation2 - inputs_sub;
    }
    else if (layer.type == NNLayer::convolution)
    {
      if (inputs_sub.dims == 4)
      {
        grad2 = activation2 - inputs_sub;
      }
      else
      {
        cv::Mat inMatSize = (cv::Mat_<int>(1, 4) << batchSize, layer.numInMaps, in_H, in_W);
        cv::Mat tmp = cv::Mat(4, (int *)inMatSize.data, CV_REAL, inputs_sub.data);
        grad2 = activation2 - tmp;
      }
    }

    // element-wise multiplication
    cv::Mat sq;
    // If grad2.dims > 2, sq.rows shall be grad2.size[0] and
    // sq.cols shall be the product of all the remaining sizes;
    elemMul(grad2, grad2, sq);

    cv::Mat sq2;
    cv::reduce(sq, sq2, 1, CV_REDUCE_SUM);  // sum up each row
    E = (real)sq2.dot(sampleWeights_sub);

    for (int i = 0; i < batchSize; ++i)
    {
      cv::Mat tmp(1, (int)grad2.total() / batchSize, grad2.type(), grad2.data + grad2.step[0] * i);
      tmp *= ((real *)sampleWeights_sub.data)[i];
    }

    // backward path : rprop

    // grad2 <- element-wise multiplication of grad2 and df2
    if (layer.type == NNLayer::perceptron)
      elemMul(df2, grad2);

    // sum up dEdw2 for all samples

    cv::Mat dEdw2;
    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
    {
      // dEdw2 <- (layer.activation^T * grad2)^T
      matMul(grad2, layer.activation, 1, dEdw2, CV_GEMM_A_T);
    }
    else if (layer.type == NNLayer::convolution)
    {
      // layer.activation is a <num samples>x<numOutMaps>x<out_H>x<out_W> matrix.
      // grad2 is a <num samples>x<numInMaps>x<in_H>x<in_W> matrix.

      // dEdw2 is a <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
      cv::Mat matSize = (cv::Mat_<int>(1, 4) << layer.numOutMaps, layer.numInMaps, filter_H, filter_W);
      dEdw2.create(4, (int *)matSize.data, CV_REAL);
      dEdw2 = 0;

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
      for (int s = 0; s < batchSize; ++s)
      {
        for (int inMap = 0; inMap < layer.numInMaps; ++inMap)
        {
          cv::Mat grad2_(in_H, in_W, CV_REAL, grad2.data + grad2.step[0] * s + grad2.step[1] * inMap);

          for (int outMap = 0; outMap < layer.numOutMaps; ++outMap)
          {
            cv::Mat activ_(out_H, out_W, CV_REAL, layer.activation.data + layer.activation.step[0] * s + layer.activation.step[1] * outMap);
            cv::Mat dEdw2_(filter_H, filter_W, CV_REAL, dEdw2.data + dEdw2.step[0] * outMap + dEdw2.step[1] * inMap);

            for (int y = 0; y < filter_H; ++y)
            {
              for (int x = 0; x < filter_W; ++x)
              {
                dEdw2_.at<real>(y, x) += (real)grad2_(cv::Rect(x, y, activ_.cols, activ_.rows)).dot(activ_);
              }
            }
          }
        }
      }
    }

    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
    {
      // layer.grad <- grad2 * layer.weight
      matMul(grad2, layer.weight, 1, layer.grad);
    }
    else if (layer.type == NNLayer::convolution)
    {
      layer.grad = 0;

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
      for (int s = 0; s < batchSize; ++s)
      {
        for (int outMap = 0; outMap < layer.numOutMaps; ++outMap)
        {
          cv::Mat grad_(out_H, out_W, CV_REAL, layer.grad.data + layer.grad.step[0] * s + layer.grad.step[1] * outMap);

          for (int inMap = 0; inMap < layer.numInMaps; ++inMap)
          {
            cv::Mat grad2_(in_H, in_W, CV_REAL, grad2.data + grad2.step[0] * s + grad2.step[1] * inMap);
            cv::Mat filter_(filter_H, filter_W, CV_REAL, layer.weight.data + layer.weight.step[0] * outMap + layer.weight.step[1] * inMap);

            cv::Ptr<cv::FilterEngine> pEngine = createLinearFilter(CV_REAL, CV_REAL, filter_);
            cv::Mat dst(in_H, in_W, CV_REAL);
            pEngine->apply(grad2_, dst);
            grad_ += dst(cv::Rect((filter_W - 1) / 2, (filter_H - 1) / 2, grad_.cols, grad_.rows));
          }
        }
      }
    }

    // update bias2
    UpdateBiasesRprop(grad2, update_param, db2, dbSign2, bias2);

    // layer 0 と　layer 1 の間の dEdw, dEdb を計算する。

    // laye.grad <- element-wise multiplication of layer.grad and layer.df
    if (layer.type == NNLayer::perceptron)
      elemMul(layer.df, layer.grad);

    // dEdw = inputs^T * layer.grad
    cv::Mat dEdw;
    if (layer.type == NNLayer::perceptron || layer.type == NNLayer::linearPerceptron)
    {
      if (inputs_sub.dims == 4)
      {
        cv::Mat tmp(batchSize, (int)dropped_inputs_sub.total() / batchSize, CV_REAL, dropped_inputs_sub.data);
        matMul(tmp, layer.grad, 1, dEdw, CV_GEMM_A_T);
      }
      else
        matMul(dropped_inputs_sub, layer.grad, 1, dEdw, CV_GEMM_A_T);
    }
    else if (layer.type == NNLayer::convolution)
    {
      // dEdw is a <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
      cv::Mat matSize = (cv::Mat_<int>(1, 4) << layer.numOutMaps, layer.numInMaps, filter_H, filter_W);
      dEdw.create(4, (int *)matSize.data, CV_REAL);
      dEdw = 0;

#if defined(_OPENMP) && defined(USE_OPENMP)
#pragma omp parallel for
#endif
      for (int s = 0; s < batchSize; ++s)
      {
        for (int outMap = 0; outMap < layer.numOutMaps; ++outMap)
        {
          cv::Mat grad_(out_H, out_W, CV_REAL, layer.grad.data + layer.grad.step[0] * s + layer.grad.step[1] * outMap);

          for (int inMap = 0; inMap < layer.numInMaps; ++inMap)
          {
            // interpret inputs as a <num samples>x<numInMaps>x<in_H>x<in_W> matrix.
            cv::Mat inputs_(in_H, in_W, CV_REAL, dropped_inputs_sub.data + dropped_inputs_sub.step[0] * s + (in_H * in_W * sizeof(real)) * inMap);
            cv::Mat dEdw_(filter_H, filter_W, CV_REAL, dEdw.data + dEdw.step[0] * outMap + dEdw.step[1] * inMap);

            for (int y = 0; y < filter_H; ++y)
            {
              for (int x = 0; x < filter_W; ++x)
              {
                dEdw_.at<real>(y, x) += (real)inputs_(cv::Rect(x, y, grad_.cols, grad_.rows)).dot(grad_);
              }
            }
          }
        }
      }
    }

    dEdw += dEdw2;

    if (update_param.type == updateParam::rprop)
    {
      // update layer.weights
      layer.UpdateWeightsRprop(dEdw, update_param);

      // update layer.bias
      UpdateBiasesRprop(layer.grad, update_param, layer.db, layer.dbSign, layer.bias);

      printf("%d. E = %lf\n", iter, double(E));
      Log("%d. E = %lf\n", iter, double(E));
    }
    else if (update_param.type == updateParam::bprop)
    {
      // update weights.

      //std::cout << std::endl << "dEdw : " << dEdw << std::endl;
      //std::cout << std::endl << "layer.last_dW : " << layer.last_dW << std::endl;
      layer.last_dW = (layer.last_dW * update_param.momentum) - (dEdw * (update_param.learningRate * (1 - update_param.momentum)));
      //std::cout << std::endl << "layer.last_dW (after update) : " << layer.last_dW << std::endl;
        
      //std::cout << std::endl << "layer.weight : " << layer.weight << std::endl;
      layer.weight += layer.last_dW;
      //std::cout << std::endl << "layer.weight (after update) : " << layer.weight << std::endl;

      // update biases.
      cv::Mat dEdb;
      cv::reduce(layer.grad, dEdb, 0, CV_REDUCE_SUM);  // sum up each column
      layer.last_db = (layer.last_db * update_param.momentum) - (dEdb * (update_param.learningRate * (1 - update_param.momentum)));
      layer.bias += layer.last_db;

      // update learning rate and momentum.
      update_param.learningRate *= update_param.learningRateDecay;
      if (update_param.learningRate < update_param.finalLearningRate)
        update_param.learningRate = update_param.finalLearningRate;

      update_param.momentum += update_param.momentumDelta;
      if (update_param.momentum > update_param.finalMomentum)
        update_param.momentum = update_param.finalMomentum;

      printf("%d. E = %f (learning rate=%f, momentum=%f)\n", iter, E, update_param.learningRate, update_param.momentum);
      Log("%d. E = %f (learning rate=%f, momentum=%f)\n", iter, E, update_param.learningRate, update_param.momentum);
    }

    if (layer.type == NNLayer::linearPerceptron && layer.maxWeightNorm != 0)
    {
      double weightNorm = cv::norm(layer.weight);
      if (layer.maxWeightNorm < weightNorm)
      {
        double coef = layer.maxWeightNorm / weightNorm;
        layer.weight *= coef;
        layer.bias *= coef;
      }
    }

    if (_access("stopLoop", 0) != -1)
    {
      DeleteFileA("stopLoop");
      break;
    }
  }

  return 0;
}

int NeuralNet::autoencode(
  const cv::Mat &inputs_,         // [num samples] x [input vector size] matrix
  const cv::Mat &sampleWeights,   // column vector of size [num samples]
  const int lastLayerToTrain,
  const updateParam &param,
  const int maxIter,
  const int minibatchSize,
  std::vector<real> &E           // out : errors.
  )
{
  const int numSamples = inputs_.rows;
  const int batchSize = (minibatchSize == 0)? numSamples : minibatchSize;

  //cv::Mat inputs = inputs_.clone(); // copy data
  cv::Mat inputs = inputs_;

  // normalize sampleWeights
  cv::Mat sumMat;
  cv::reduce(sampleWeights, sumMat, 0, CV_REDUCE_SUM);  // sum up the only column
  sampleWeights /= sumMat.at<real>(0, 0);

  E.resize(lastLayerToTrain + 1);
  for (int layer = 0; layer <= lastLayerToTrain; ++layer)
  {
    printf("Layer %d\n", layer);
    LogA("Layer %d\n", layer);

    int ret = autoencode_one_layer(layer, inputs, sampleWeights, param, maxIter, minibatchSize, E[layer]);
    if (ret == 8)
    {
      if (layers[layer].type == NNLayer::maxPool)
      {
        cv::Mat matSize = (cv::Mat_<int>(1, 4) << batchSize, layers[layer].numOutMaps, layers[layer].outMapSize.at<int>(0), layers[layer].outMapSize.at<int>(1));
        layers[layer].activation.create(4, (int *)matSize.data, CV_REAL);
        layers[layer].df.create(4, (int *)matSize.data, CV_32S);
        if (layers[layer].activation.data == 0 || layers[layer].df.data == 0)
          return 1;
      }

      const cv::Mat *pX = &inputs;
      layers[layer].forwardPropagate(pX, false);
    }
    else if (ret != 0)
    {
      Log("Error: autoencode_one_layer() returned %d.\n", ret);
      return 2;
    }

    inputs = layers[layer].activation;
  }

  return 0;
}

int NeuralNet::predict(const cv::Mat &inputs, cv::Mat &outputs, const int minibatchSize)
{
  const int numLayers = (int)layers.size();

  if (layers[0].type == NNLayer::maxPool)
  {
    layers[0].inMapSize.at<int>(1) = inputs.cols;
    if (inputs.cols % layers[0].filterSize.at<int>(1) != 0)
      return 1;
    layers[0].outMapSize.at<int>(1) = inputs.cols / layers[0].filterSize.at<int>(1);
  }

  if (layers[0].type == NNLayer::perceptron || layers[0].type == NNLayer::linearPerceptron)
  {
    if (inputs.cols != layers[0].inSize)
    {
      // input size mismatch.
      return 2;
    }
  }
  else if (layers[0].type == NNLayer::convolution)
  {
    if (inputs.cols != layers[0].inMapSize.at<int>(0) * layers[0].inMapSize.at<int>(0))
      // input size mismatch.
      return 3;
    else if (layers[0].numInMaps != 1)
      // number of input feature maps of layer 0 should be 1
      return 4;
  }
  else if (
    layers[numLayers - 1].type != NNLayer::perceptron &&
    layers[numLayers - 1].type != NNLayer::linearPerceptron &&
    layers[numLayers - 1].type != NNLayer::softMax)
  {
    return 5;
  }

  const int numSamples = inputs.size[0];
  const int batchSize = (minibatchSize == 0)? numSamples : minibatchSize;
  outputs.create(numSamples, layers[numLayers - 1].outSize, CV_REAL);

  // この関数は funcToEvaluateEvery() を通して NeuralNet::train() の内部からも呼ばれることがある。
  // すべてのレイヤーの activation と df のサイズを復元できるよう、保存しておく。
  std::vector<cv::Mat> orig_activation(numLayers);
  std::vector<cv::Mat> orig_df(numLayers);

  for (int i = 0; i < numLayers; ++i)
  {
    // 保存
    orig_activation[i] = layers[i].activation;
    orig_df[i] = layers[i].df;

    if (layers[i].type == NNLayer::perceptron)
    {
      layers[i].activation.create(batchSize, layers[i].outSize, CV_REAL);
      layers[i].df.create(batchSize, layers[i].outSize, CV_REAL);

      if (layers[i].df.data == 0)
        return 6;
    }
    else if (layers[i].type == NNLayer::linearPerceptron)
    {
      layers[i].activation.create(batchSize, layers[i].outSize, CV_REAL);
    }
    else if (layers[i].type == NNLayer::softMax)
    {
      layers[i].activation.create(batchSize, layers[i].outSize, CV_REAL);
    }
    else if (layers[i].type == NNLayer::convolution)
    {
      // <num samples>x<numOutMaps>x<out_H>x<out_W> matrix
      cv::Mat matSize = (cv::Mat_<int>(1, 4) << batchSize, layers[i].numOutMaps, layers[i].outMapSize.at<int>(0), layers[i].outMapSize.at<int>(1));
      layers[i].activation.create(4, (int *)matSize.data, CV_REAL);
      layers[i].df.create(4, (int *)matSize.data, CV_REAL);
      if (layers[i].df.data == 0)
        return 7;
    }
    else if (layers[i].type == NNLayer::maxPool)
    {
      // <num samples>x<numOutMaps>x<out_H>x<out_W> matrix
      cv::Mat matSize = (cv::Mat_<int>(1, 4) << batchSize, layers[i].numOutMaps, layers[i].outMapSize.at<int>(0), layers[i].outMapSize.at<int>(1));
      layers[i].activation.create(4, (int *)matSize.data, CV_REAL);
      layers[i].df.create(4, (int *)matSize.data, CV_32S);
      if (layers[i].df.data == 0)
        return 8;
    }

    if (layers[i].activation.data == 0)
      return 9;
  }

  // forward pass
  if (minibatchSize == 0)
  {
    const cv::Mat *pX = &inputs;
    for (int i = 0; i < numLayers; ++i)
      layers[i].forwardPropagate(pX, false);

    pX->copyTo(outputs);
  }
  else
  {
    // set identity permutation
    std::vector<unsigned int> permutation(numSamples);
    for (int i = 0; i < numSamples; ++i)
      permutation[i] = i;

    cv::Mat inputs_sub;
    int sampleHead = 0; // the index of the first sample in the current minibatch
    for (int sampleHead = 0; sampleHead < numSamples; sampleHead += minibatchSize)
    {
      ChooseSamples(inputs, inputs_sub, sampleHead, minibatchSize, permutation);

      const cv::Mat *pX = &inputs_sub;
      for (int i = 0; i < numLayers; ++i)
        layers[i].forwardPropagate(pX, false);

      int numOutRows = (sampleHead + minibatchSize < numSamples)? minibatchSize : (numSamples - sampleHead);
      pX->rowRange(0, numOutRows).copyTo(outputs.rowRange(sampleHead, sampleHead + numOutRows));
    }
  }

  for (int i = 0; i < numLayers; ++i)
  {
    // 復元
    layers[i].activation = orig_activation[i];
    layers[i].df = orig_df[i];
  }

  return 0;
}

int NeuralNet::writeBinary(const char* filename) const
{
  FILE *fp;
  errno_t errnum = fopen_s(&fp, filename, "wb");
  if (errnum != 0)
    return errnum;

  fwrite(&firstCharCode, sizeof(firstCharCode), 1, fp);

  const int numLayers = (int)layers.size();
  fwrite(&numLayers, sizeof(numLayers), 1, fp);
  for (int i = 0; i < numLayers; ++i)
    layers[i].writeBinary(fp);

  fclose(fp);

  return 0;
}

int NeuralNet::readBinary(const char* filename/*, std::vector<NNLayer> &layers*/)
{
  FILE *fp;
  errno_t errnum = fopen_s(&fp, filename, "rb");
  if (errnum != 0)
    return errnum;

  fread(&firstCharCode, sizeof(firstCharCode), 1, fp);

  int numLayers;
  fread(&numLayers, sizeof(numLayers), 1, fp);
  layers.resize(numLayers);
  for (int i = 0; i < numLayers; ++i)
    layers[i].readBinary(fp);

  fclose(fp);

  return 0;
}

// For input_i of the neural net, compute its contribution to the layer's output_j (= d output_j/d input_i)
// for all i and j, and make a image out of them.
int NeuralNet::writeWeightImage(
  int layer,             // layer to handle.
  const char *fileName,  // output file name
  int cellWidth,         // cell width
  bool cumulative,       // calculate cumulative weight (for perceptron layers)
  real coef              // output is multiplied by this coefficient to emphasize
  ) const
{
  const int numLayers = (int)layers.size();

  if (layer < 0 || numLayers <= layer)
    return 1;

  if (
    layers[layer].type != NNLayer::perceptron &&
    layers[layer].type != NNLayer::linearPerceptron &&
    layers[layer].type != NNLayer::convolution)
    return 2;

  if (layers[layer].type == NNLayer::perceptron || layers[layer].type == NNLayer::linearPerceptron)
  {
    cv::Mat W;
    if (!cumulative || layer == 0)
      W = layers[layer].weight;
    else
    {
      cv::Mat tmp1, tmp2;

      if (layers[0].type != NNLayer::perceptron && layers[0].type != NNLayer::linearPerceptron)
        return 3;

      tmp1 = layers[0].weight;
      int i;
      for (i = 1; i <= layer; ++i)
      {
        if (layers[i].type == NNLayer::perceptron || layers[i].type == NNLayer::linearPerceptron)
        {
          if ((i % 2) == 1)
            // tmp2 <- tmp1 * weight_i
            matMul(tmp1, layers[i].weight, 1, tmp2);
          else
            // tmp1 <- tmp2 * weight_i
            matMul(tmp2, layers[i].weight, 1, tmp1);
        }
        else if (layers[i].type == NNLayer::maxPool)
        {
          if ((i % 2) == 1)
          {
            // tmp2 <- tmp1 の列を filterSize.at<int>(1) ずつ足し合わせて平均したもの
            const int newCols = tmp1.cols / layers[i].filterSize.at<int>(1);
            tmp2.create(tmp1.rows, newCols, tmp1.type());
            tmp2 = 0;
            for (int col = 0; col < newCols; ++col)
            {
              for (int j = 0; j < layers[i].filterSize.at<int>(1); ++j)
                tmp2.col(col) += tmp1.col(col*2+j);
            }
            tmp2 /= layers[i].filterSize.at<int>(1);
          }
          else
          {
            // tmp1 <- tmp2 の列を filterSize.at<int>(1) ずつ足し合わせて平均したもの
            const int newCols = tmp2.cols / layers[i].filterSize.at<int>(1);
            tmp1.create(tmp2.rows, newCols, tmp2.type());
            tmp1 = 0;
            for (int col = 0; col < newCols; ++col)
            {
              for (int j = 0; j < layers[i].filterSize.at<int>(1); ++j)
                tmp1.col(col) += tmp2.col(col*2+j);
            }
            tmp1 /= layers[i].filterSize.at<int>(1);
          }
        }
        else
          return 4;
      }
      W = ((i % 2) == 1)? tmp1 : tmp2;
    }

    if (cellWidth == 0)
    {
      cellWidth = (int)sqrt(W.rows);
      while ((W.rows % cellWidth) != 0)
        --cellWidth;
    }

    const int numInputs = W.rows;
    const int numOutputs = W.cols;
    const int cellHeight = (numInputs - 1) / cellWidth + 1;
    const int numCellsX = (int)sqrt((real)numOutputs);
    const int numCellsY = (numOutputs - 1) / numCellsX + 1;

    if (coef == 0)
    {
      cv::Scalar mean, stddev;
      meanStdDev(W, mean, stddev);
      coef = 10 / (real)stddev[0];
    }

    cv::Mat subMat = W * coef;
    exp(-subMat, subMat);
    subMat = 1 / (1 + subMat);
    subMat.convertTo(subMat, CV_8U, 256);

    // 一画像が一行に対応するよう、転置する。
    subMat = subMat.t();

    cv::Mat img(numCellsY * (cellHeight + 1) - 1, numCellsX * (cellWidth + 1) - 1, CV_8U, cv::Scalar(0));
    for (int cell_y = 0; cell_y < numCellsY; ++cell_y)
    {
      for (int cell_x = 0; cell_x < numCellsX; ++cell_x)
      {
        int index2 = cell_y * numCellsX + cell_x;
        if (index2 >= numOutputs)
          break;

        subMat.row(index2).reshape(0, cellHeight).copyTo(img(
          cv::Range((cellHeight + 1) * cell_y, (cellHeight + 1) * (cell_y + 1) - 1),
          cv::Range((cellWidth + 1) * cell_x, (cellWidth + 1) * (cell_x + 1) - 1)));
      }
    }

    cv::imwrite(fileName, img);  
  }
  else if (layers[layer].type == NNLayer::convolution)
  {
    const int cellHeight = layers[layer].filterSize.at<int>(0);
    const int cellWidth = layers[layer].filterSize.at<int>(1);
    const int numCellsX = layers[layer].numInMaps;
    const int numCellsY = layers[layer].numOutMaps;

    cv::Mat img(numCellsY * (cellHeight + 1) - 1, numCellsX * (cellWidth + 1) - 1, CV_8U, cv::Scalar(0));
    cv::Mat tmp;

    for (int cell_y = 0; cell_y < numCellsY; ++cell_y)
    {
      for (int cell_x = 0; cell_x < numCellsX; ++cell_x)
      {
        int index2 = cell_y * numCellsX + cell_x;

        // weight is <numOutMaps>x<numInMaps>x<filter_H>x<filter_W> matrix.
        cv::Mat filter(cellHeight, cellWidth, CV_REAL, layers[layer].weight.data + layers[layer].weight.step[0] * cell_y + layers[layer].weight.step[1] * cell_x);

        if (coef == (real)0)
        {
          cv::Scalar mean, stddev;
          meanStdDev(filter, mean, stddev);
          tmp = (filter - mean[0]) * (1 / stddev[0]);
        }
        else
          tmp = filter * coef;
        cv::exp(-tmp, tmp);
        tmp = 256 / (1 + tmp);
        tmp.convertTo(tmp, CV_8U);

        tmp.copyTo(img(cv::Rect((cellWidth + 1) * cell_x, (cellHeight + 1) * cell_y, cellWidth, cellHeight)));
      }
    }

    cv::imwrite(fileName, img);  
  }

  return 0;
}


