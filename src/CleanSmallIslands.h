/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

#pragma once

// cleanSmallIslands()
// lum1 以下の点からなる島を見つけ、島の中に lum2 以下の点が numBlackThreshold 以下しかなければ、その島を白くする。
// targetImage : 変更を書き込む画像
// refImage : 参照画像
//
// param: lum1, lum2, numBlackThreshold
//
// phase 1:
// すべてのピクセルに上から順、左から順に番号(int)をつけていく。
// lum > lum1 なら 0, lum <= lum1 ならそのような連続するピクセルに同じ番号をつける。
// 0 以外の番号をつける場合、一つ上の行に 0 以外の点があれば、その番号を踏襲する。
// また、一つ上の行に 2 種類以上の番号が隣接していれば、それらの番号は統合される。
// y方向に進むにつれて番号は統合されていく。
// 00000011100000020000000000030000
// 00000111100022222000000000333000
// 00001111111111111100000003333300 <- ここで 1 と 2 は統合。
//
// phase 2:
// それぞれの島(番号)について、lum2 以下のピクセルの数が numBlackThreshold 以下ならば、その島を lum1 で塗りつぶす。
//
// phase 3: 
// lum1 以上の点をすべて白(255)にする。
void CleanSmallIslands(cv::Mat &targetImage, const cv::Mat &refImage,
					   const unsigned char lum1, const unsigned char lum2, const int numBlackThreshold,
					   const bool cleanIslandsTouchingOuterEdge);

