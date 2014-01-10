/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

#pragma once

// cleanSmallIslands()
// lum1 �ȉ��̓_����Ȃ铇�������A���̒��� lum2 �ȉ��̓_�� numBlackThreshold �ȉ������Ȃ���΁A���̓��𔒂�����B
// targetImage : �ύX���������މ摜
// refImage : �Q�Ɖ摜
//
// param: lum1, lum2, numBlackThreshold
//
// phase 1:
// ���ׂẴs�N�Z���ɏォ�珇�A�����珇�ɔԍ�(int)�����Ă����B
// lum > lum1 �Ȃ� 0, lum <= lum1 �Ȃ炻�̂悤�ȘA������s�N�Z���ɓ����ԍ�������B
// 0 �ȊO�̔ԍ�������ꍇ�A���̍s�� 0 �ȊO�̓_������΁A���̔ԍ��𓥏P����B
// �܂��A���̍s�� 2 ��ވȏ�̔ԍ����אڂ��Ă���΁A�����̔ԍ��͓��������B
// y�����ɐi�ނɂ�Ĕԍ��͓�������Ă����B
// 00000011100000020000000000030000
// 00000111100022222000000000333000
// 00001111111111111100000003333300 <- ������ 1 �� 2 �͓����B
//
// phase 2:
// ���ꂼ��̓�(�ԍ�)�ɂ��āAlum2 �ȉ��̃s�N�Z���̐��� numBlackThreshold �ȉ��Ȃ�΁A���̓��� lum1 �œh��Ԃ��B
//
// phase 3: 
// lum1 �ȏ�̓_�����ׂĔ�(255)�ɂ���B
void CleanSmallIslands(cv::Mat &targetImage, const cv::Mat &refImage,
					   const unsigned char lum1, const unsigned char lum2, const int numBlackThreshold,
					   const bool cleanIslandsTouchingOuterEdge);

