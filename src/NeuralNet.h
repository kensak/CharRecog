/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

#pragma once

#ifndef __OPENCV_NEURALNET_H__
#define __OPENCV_NEURALNET_H__

#include "config.h"
#include <opencv2/opencv.hpp>

#ifdef REAL_IS_FLOAT
// float î≈
typedef float real;
#define CV_REAL CV_32F
#define REAL_EPSILON FLT_EPSILON
const char realStr[] = "float";
#else
// double î≈
typedef double real;
#define CV_REAL CV_64F
#define REAL_EPSILON DBL_EPSILON
const char realStr[] = "double";
#endif

struct updateParam
{
  enum {bprop, rprop} type;

  // bprop (standard backprop)
  real learningRate;       // learning rate
  real finalLearningRate;  // learning rate ÇÃç≈èIíl
  real learningRateDecay;  // åJÇËï‘ÇµñàÇ… learning rate Ç…ä|ÇØÇÈåWêî
  real momentum;           // momentum
  real initMomentum;       // momentum ÇÃèâä˙íl
  real finalMomentum;      // momentum ÇÃç≈èIíl
  int momentumDecayEpoch;  // Ç±ÇÃâÒêîå„ÇÕ momentum ÇÕ finalMomentum ÇÃÇ‹Ç‹Ç…Ç»ÇÈÅB

  // rprop
  real dw0;
  real dw_plus;
  real dw_minus;
  real dw_max;
  real dw_min;

  updateParam() : type(rprop),
    learningRate((real)0.1), finalLearningRate((real)0.00001), learningRateDecay((real)0.998),
    momentum((real)0), initMomentum((real)0.0), finalMomentum((real)0.9), momentumDecayEpoch(200),
    dw0((real)0.001), dw_plus((real)1.2), dw_minus((real)0.5), dw_max((real)50.0), dw_min(REAL_EPSILON) {}
};

class NNLayer
{
  friend class NeuralNet;

  enum layerType {none, perceptron, linearPerceptron, convolution, maxPool, softMax};

  layerType type;

  // for perceptron, linearPerceptron and softMax ------------------------------------------------
  int inSize;
  int outSize;
  real dropoutRatio;  // Ratio of 'dropped-out' inputs.
                      // For autoencoder, this is the corruption level of the inputs.

  cv::Mat weight;     // inSize * outSize matrix
  cv::Mat bias;       // row vector of length outSize

  cv::Mat df;         // f'(y) : numSamples * outSize matrix
                      // For Max Pool layer, this is the position of the activated neuron.
  cv::Mat activation; // f(y)  : numSamples * outSize matrix
  cv::Mat grad;       // dE/dy : numSamples * outSize matrix

  // for linearPerceptron ------------------------------------------------------------------------
  real maxWeightNorm; // The max value of ||weight||_2. 
                      // If, as a result of an update, the L_2 norm of 'weight'
                      // exceeds 'maxWeightNorm', 'weight' is scaled down so as to
                      // make it have a L_2 norm of 'maxWeightNorm'.
                      // If 0, no scaling down will occur.

  // for convolution and maxPool -----------------------------------------------------------------
  cv::Mat inMapSize;  // e.g. {13, 13} for input feature maps of size HxW=13x13
  cv::Mat filterSize; // e.g. {5, 5} for HxW=5x5 filters.
  cv::Mat outMapSize; // e.g. {9, 9} for output feature maps of size HxW=9x9 (9=13-5+1)
  int numInMaps;
  int numOutMaps;

  // training method -----------------------------------------------------------------------------

  // for rprop
  cv::Mat dw;         // delta_W (absolute value)
  cv::Mat dwSign;
  cv::Mat db;         // delta_bias (absolute value)
  cv::Mat dbSign;     

  // for backprop
  cv::Mat last_dW;    // memorize the last delta_W
  cv::Mat last_db;    // memorize the last delta_bias

  //----------------------------------------------------------------------------------------------

  void forwardPropagate(const cv::Mat* &pX, const bool dropout = false);
  void activateTanh(const cv::Mat &y);
  void activateSoftMax(const cv::Mat &y);
  void UpdateWeightsRprop(const cv::Mat &dEdw, const updateParam &param);

  void writeBinary(FILE *fp) const;
  void readBinary(FILE *fp);

public:

  NNLayer() : type(none), dropoutRatio(0), maxWeightNorm(0) {}

  void logSettings() const;

  // Create perceptron layer whose activation function is 1.7159 * tanh(y * 2 / 3).
  void createPerceptronLayer(const int inSize, const int outSize, const real dropoutRatio = 0);

  // Create perceptron layer whose activation function is the identity function.
  void createLinearPerceptronLayer(const int inSize, const int outSize, const real dropoutRatio = 0, const real maxWeightNorm = 0);

  void createConvolutionLayer(
    const cv::Mat inMapSize,     // e.g. {13, 13} for input feature maps of size HxW=13x13
    //const cv::Mat filterSizez  // e.g. {40, 20, 5, 5} for 40 output maps, 20 input maps, HxW=5x5 filters. 
    const cv::Mat filterSize,    // e.g. {5, 5} for HxW=5x5 filters.
    const int numInMaps,         // number of input feature maps
    const int numOutMaps         // number of output feature maps
    );
  // With above parameters, this layer outputs 40 maps of size 9x9 (9=13-5+1), out of 20 maps of size 13x13,
  // using 800 (=20*40) filters of size 5x5.

  // Create 2D max-pool layer.
  void createMaxPoolLayer(
    //const cv::Mat inSizes,     // e.g. {20, 26, 26} for 20 maps of size HxW=26x26
    const cv::Mat filterSize     // e.g. {2, 2} for HxW=2x2 filters. 
    );
  // With above parameters, this layer outputs 20 maps of size 13x13 (13=26/2), out of 20 maps of size 26x26,
  // using the max-pool filter of size 2x2.

  void createSoftMaxLayer(const int inOutSize);

};

struct GPU_info_type
{
  GPU_info_type() : GPU_exists(false), supportsDouble(false), limitedDouble(false) {}

  std::wstring description;
  std::wstring device_path;
  bool GPU_exists;
  bool supportsDouble;
  bool limitedDouble;
};

class NeuralNet
{
  std::vector<NNLayer> layers;
  GPU_info_type GPU_info;

  int constructLayers(const std::string &layerParamStr);

  /*
  int train_backprop(
    const cv::Mat &inputs,         // [num samples] x [input vector size] matrix
    const cv::Mat &outputs,        // [num samples] x [output vector size] matrix
    const cv::Mat &sampleWeights,  // column vector of size [num samples]
    const real learningRate,
    real &E                        // out : error.
    );
    */

  int train_sub(
    const cv::Mat &inputs,         // [num samples] x [input vector size] matrix
    const cv::Mat &outputs,        // [num samples] x [output vector size] matrix
    const cv::Mat &sampleWeights,  // column vector of size [num samples]
    const int firstLayerToTrain,   // index of the first layer to train
    const updateParam &update_param,
    real &E                        // out : error.
    );

  int autoencode_one_layer(
    const int layer,
    const cv::Mat &inputs,          // [num samples] x [input vector size] matrix
    const cv::Mat &sampleWeights,   // column vector of size [num samples]
    const updateParam &update_param,
    const int maxIter,
    real &E                         // out : error.
    );

public:

  //enum backPropType {backprop, rprop};
  unsigned _int16 firstCharCode;

  NeuralNet();

  int create(const std::string &layerParamStr/*std::vector<NNLayer> &layers*/);

  void logSettings() const;
  inline int numLayers() const { return (int)layers.size(); }
  inline int outSize() const { return layers[numLayers() - 1].outSize; }

  int train(
    const cv::Mat &inputs,         // [num samples] x [input vector size] matrix
    void (*TransformSamples)(cv::Mat &inputs, void *transformSamplesInfo), // callback function for transforming input data
    void *transformSamplesInfo,    // extra info passed to TransformSamples()
    const cv::Mat &outputs,        // [num samples] x [output vector size] matrix
    const cv::Mat &sampleWeights,  // column vector of size [num samples]
    const int firstLayerToTrain,
    const updateParam &update_param,
    const int maxIter,
    const int evaluateEvery,
    void (*funcToEvaluateEvery)(NeuralNet &nn), // callback function to be called every 'evaluateEvery' epochs
    real &E                        // out : error.
    );

  int autoencode(
    const cv::Mat& inputs,         // [num samples] x [input vector size] matrix
    const cv::Mat& sampleWeights,  // column vector of size [num samples]
    const int lastLayerToTrain,
    const updateParam &update_param,
    const int maxIter,
    std::vector<real> &E           // out : errors.
    );

  int predict(const cv::Mat &inputs, cv::Mat &outputs);

  int writeBinary(const char* filename) const;
  int readBinary(const char* filename);

  // For input_i of the neural net, compute its contribution to the layer's output_j (= d output_j/d input_i)
  // for all i and j, and make a image out of them.

  // writes out weights to an image file.
  //
  // If cumulative=true, calculates ÉÆ_{i=0~layer}W_i.
  int writeWeightImage(
    int layer,               // layer to handle.
    const char *fileName,    // output file name
    int cellWidth = 0,       // cell width (for perceptron layer)
    bool cumulative = false, // calculate cumulative weight (only for perceptron layer)
    real coef = 0            // output is multiplied by this coefficient to emphasize
    ) const;

};


#endif