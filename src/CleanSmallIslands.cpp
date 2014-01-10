/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "fArray.h"
#include "dict.h"


class setWithRepresentative
{
	fArray<unsigned int> representative;    // representative[elem] は elem を含む集合の代表。elem がどの集合にも属していない場合、representative[elem] は不定。
	fArray<fArray<unsigned int> > elements; // elements[repr] は repr を代表とする集合の repr 以外の要素のリスト。

public:

	setWithRepresentative( const unsigned int initialLength ) : representative( 0, initialLength ), elements( 0, initialLength ) {}

	// elem を含む集合の代表を返す。
	inline unsigned int representativeOf( const unsigned int elem ) const
	{
		return representative[elem];
	}

	// elem1 と elem2 が同じ集合の元であれば true を返す。
	inline bool inSameSet( const unsigned int elem1, const unsigned int elem2 ) const
	{
		return( representative[elem1] == representative[elem2] );
	}

	// elem1 を含む集合と elem2 を含む集合を合併する。
	// elem1 の代表が新たな集合の代表となる。
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

	// elem1 を含む集合に新たな要素 elem2 を追加する。
	void addNewElem( const unsigned int elem1, const unsigned int elem2 )
	{
		unsigned int repr1 = representative[elem1];
		fArray<unsigned int> &elems1 = elements(repr1);
		elems1.append(elem2);
		representative(elem2) = repr1;
	}

	// elem からのみなる集合を新たに作る。
	inline void createNewSet( const unsigned int elem )
	{
		representative(elem) = elem;
	}
};

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
	dictionary<unsigned int, int> numPix( 0 ); // lum1 以上の点の数。
	dictionary<unsigned int, int> numPix2( 0 ); // lum2 以上の点の数。
	fArray<unsigned int> islandsTouchingOuterEdge; // image の外周に接している島の idx のリスト。

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
			if( refImage.ptr()[p] <= lum1 ) // 黒い点
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

				if( 0 < y && (upperIdx = islandIndex[p-width]) != 0 ) // 上に黒いピクセルがある
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
				else // 上のピクセルは白
				{
					firstIdxU = true;
				}

				islandIndex[p] = currentIdx;
				unsigned int reprC = idxSet.representativeOf( currentIdx );
				numPix(reprC)++;
				if( refImage.ptr()[p] <= lum2 )
					numPix2(reprC)++;
			}
			else // 白い点
			{
				currentIdx = 0;
				firstIdxU = true;
			}
		}
	}

	dictionary<unsigned int, bool> isTouchingOuterEdge( false ); // repr が代表する島が外周に接していれば、 isTouchingOuterEdge[repr] は true。
	if( cleanIslandsTouchingOuterEdge )
	{
		// 集合の要素が一つでも isTouchingOuterEdge[elem]==true なら、代表元も ture にする。
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

