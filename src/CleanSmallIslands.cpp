/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "fArray.h"
#include "dict.h"


class setWithRepresentative
{
	fArray<unsigned int> representative;    // representative[elem] �� elem ���܂ޏW���̑�\�Belem ���ǂ̏W���ɂ������Ă��Ȃ��ꍇ�Arepresentative[elem] �͕s��B
	fArray<fArray<unsigned int> > elements; // elements[repr] �� repr ���\�Ƃ���W���� repr �ȊO�̗v�f�̃��X�g�B

public:

	setWithRepresentative( const unsigned int initialLength ) : representative( 0, initialLength ), elements( 0, initialLength ) {}

	// elem ���܂ޏW���̑�\��Ԃ��B
	inline unsigned int representativeOf( const unsigned int elem ) const
	{
		return representative[elem];
	}

	// elem1 �� elem2 �������W���̌��ł���� true ��Ԃ��B
	inline bool inSameSet( const unsigned int elem1, const unsigned int elem2 ) const
	{
		return( representative[elem1] == representative[elem2] );
	}

	// elem1 ���܂ޏW���� elem2 ���܂ޏW������������B
	// elem1 �̑�\���V���ȏW���̑�\�ƂȂ�B
	void join( const unsigned int elem1, const unsigned int elem2 )
	{
		unsigned int repr1 = representative[elem1];
		unsigned int repr2 = representative[elem2];

		if( repr1 == repr2 )
			return;

		elements( repr2 ); // resize if neccesary.
		fArray<unsigned int> &elems1 = elements(repr1);
		fArray<unsigned int> &elems2 = elements(repr2);

		for( int i = 0; i < (int)elems2.length(); ++i )
		{
			elems1.append(elems2[i]);
			representative[elems2[i]] = repr1;
		}
		elems1.append(repr2);
		representative[repr2] = repr1;

		elems2.setLength( 0 );
	}

	// elem1 ���܂ޏW���ɐV���ȗv�f elem2 ��ǉ�����B
	void addNewElem( const unsigned int elem1, const unsigned int elem2 )
	{
		unsigned int repr1 = representative[elem1];
		fArray<unsigned int> &elems1 = elements(repr1);
		elems1.append(elem2);
		representative(elem2) = repr1;
	}

	// elem ����݂̂Ȃ�W����V���ɍ��B
	inline void createNewSet( const unsigned int elem )
	{
		representative(elem) = elem;
	}
};

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
					   const bool cleanIslandsTouchingOuterEdge)
{
    CV_Assert(refImage.type() == CV_8UC1);

	targetImage.create(refImage.rows, refImage.cols, refImage.type());

	if (refImage.data == NULL)
		return;

	int height = refImage.rows;
	int width = refImage.cols;

	fArray<unsigned int> islandIndex( height*width );
	memset( islandIndex.rawData(), 0, height*width*sizeof(int) );

	int y, p = 0;
	unsigned int upperIdx, currentIdx = 0, newIdx = 1;
	bool firstIdxU = true;
	setWithRepresentative idxSet( 512 );
	dictionary<unsigned int, int> numPix( 0 ); // lum1 �ȏ�̓_�̐��B
	dictionary<unsigned int, int> numPix2( 0 ); // lum2 �ȏ�̓_�̐��B
	fArray<unsigned int> islandsTouchingOuterEdge; // image �̊O���ɐڂ��Ă��铇�� idx �̃��X�g�B

	// phase 1
	for( y = 0; y < height; ++y )
	{
		if( currentIdx != 0 )
		{
			islandsTouchingOuterEdge.append( currentIdx );
			currentIdx = 0;
		}

		firstIdxU = true;

		for( int x = 0; x < width; ++x, ++p )
		{
			if( refImage.ptr()[p] <= lum1 ) // �����_
			{
				if( currentIdx == 0 )
				{
					currentIdx = newIdx;
					++newIdx;
					idxSet.createNewSet( currentIdx );

					if( y == 0 || y == height-1 || x == 0 )
					{
						islandsTouchingOuterEdge.append( currentIdx );
					}
				}

				if( 0 < y && (upperIdx = islandIndex[p-width]) != 0 ) // ��ɍ����s�N�Z��������
				{
					if( firstIdxU )
					{
						unsigned int reprU = idxSet.representativeOf( upperIdx );
						unsigned int reprC = idxSet.representativeOf( currentIdx );

						if( reprU != reprC )
						{
							idxSet.join( upperIdx, currentIdx );
							numPix(reprU) += numPix[reprC];
							numPix2(reprU) += numPix2[reprC];
						}
						
						firstIdxU = false;
					}
				}
				else // ��̃s�N�Z���͔�
				{
					firstIdxU = true;
				}

				islandIndex[p] = currentIdx;
				unsigned int reprC = idxSet.representativeOf( currentIdx );
				numPix(reprC)++;
				if( refImage.ptr()[p] <= lum2 )
					numPix2(reprC)++;
			}
			else // �����_
			{
				currentIdx = 0;
				firstIdxU = true;
			}
		}
	}

	dictionary<unsigned int, bool> isTouchingOuterEdge( false ); // repr ����\���铇���O���ɐڂ��Ă���΁A isTouchingOuterEdge[repr] �� true�B
	if( cleanIslandsTouchingOuterEdge )
	{
		// �W���̗v�f����ł� isTouchingOuterEdge[elem]==true �Ȃ�A��\���� ture �ɂ���B
		for( int i = 0; i < (int)islandsTouchingOuterEdge.length(); ++i )
		{
			isTouchingOuterEdge(idxSet.representativeOf(islandsTouchingOuterEdge[i])) = true;
		}
	}

	// phase 2, 3
	p = 0;
	for( y = 0; y < height; ++y )
	{
		for( int x = 0; x < width; ++x, ++p )
		{
			unsigned int idx = islandIndex[p];
			if( idx == 0 )
			{
				if( lum1 <= refImage.ptr()[p] )
					targetImage.ptr()[p] = 255;
			}
			else
			{
				if( numPix2[idxSet.representativeOf(idx)] <= numBlackThreshold || cleanIslandsTouchingOuterEdge && isTouchingOuterEdge[idxSet.representativeOf(idx)] )
					targetImage.ptr()[p] = 255;
			}
		}
	}
}

