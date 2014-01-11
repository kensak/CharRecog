/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

#pragma once

#include <opencv2/opencv.hpp>

namespace cv
{
  Mat reshape(const Mat &mat, int newndims, const int* newsz);
}