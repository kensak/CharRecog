/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

#include "stdafx.h"
#include <opencv2/opencv.hpp>

namespace cv {

static inline void setSize( Mat& m, int _dims, const int* _sz,
                            const size_t* _steps, bool autoSteps=false )
{
    CV_Assert( 0 <= _dims && _dims <= CV_MAX_DIM );
    if( m.dims != _dims )
    {
        if( m.step.p != m.step.buf )
        {
            fastFree(m.step.p);
            m.step.p = m.step.buf;
            m.size.p = &m.rows;
        }
        if( _dims > 2 )
        {
            m.step.p = (size_t*)fastMalloc(_dims*sizeof(m.step.p[0]) + (_dims+1)*sizeof(m.size.p[0]));
            m.size.p = (int*)(m.step.p + _dims) + 1;
            m.size.p[-1] = _dims;
            m.rows = m.cols = -1;
        }
    }

    m.dims = _dims;
    if( !_sz )
        return;

    size_t esz = CV_ELEM_SIZE(m.flags), total = esz;
    int i;
    for( i = _dims-1; i >= 0; i-- )
    {
        int s = _sz[i];
        CV_Assert( s >= 0 );
        m.size.p[i] = s;

        if( _steps )
            m.step.p[i] = i < _dims-1 ? _steps[i] : esz;
        else if( autoSteps )
        {
            m.step.p[i] = total;
            int64 total1 = (int64)total*s;
            if( (uint64)total1 != (size_t)total1 )
                CV_Error( CV_StsOutOfRange, "The total matrix size does not fit to \"size_t\" type" );
            total = (size_t)total1;
        }
    }

    if( _dims == 1 )
    {
        m.dims = 2;
        m.cols = 1;
        m.step[1] = esz;
    }
}

static void updateContinuityFlag(Mat& m)
{
    int i, j;
    for( i = 0; i < m.dims; i++ )
    {
        if( m.size[i] > 1 )
            break;
    }

    for( j = m.dims-1; j > i; j-- )
    {
        if( m.step[j]*m.size[j] < m.step[j-1] )
            break;
    }

    uint64 t = (uint64)m.step[0]*m.size[0];
    if( j <= i && t == (size_t)t )
        m.flags |= Mat::CONTINUOUS_FLAG;
    else
        m.flags &= ~Mat::CONTINUOUS_FLAG;
}

static void finalizeHdr(Mat& m)
{
    updateContinuityFlag(m);
    int d = m.dims;
    if( d > 2 )
        m.rows = m.cols = -1;
    if( m.data )
    {
        m.datalimit = m.datastart + m.size[0]*m.step[0];
        if( m.size[0] > 0 )
        {
            m.dataend = m.data + m.size[d-1]*m.step[d-1];
            for( int i = 0; i < d-1; i++ )
                m.dataend += (m.size[i] - 1)*m.step[i];
        }
        else
            m.dataend = m.datalimit;
    }
    else
        m.dataend = m.datalimit = 0;
}

Mat reshape(const Mat &mat, int newndims, const int* newsz)
{
  CV_Assert(mat.channels() == 1);

  Mat hdr = mat;
  setSize(hdr, newndims, newsz, 0, true);
  finalizeHdr(hdr);

  return hdr;
}

/*
void Mat::create(int d, const int* _sizes, int _type)
{
    int i;
    CV_Assert(0 <= d && _sizes && d <= CV_MAX_DIM && _sizes);
    _type = CV_MAT_TYPE(_type);

    if( data && (d == dims || (d == 1 && dims <= 2)) && _type == type() )
    {
        if( d == 2 && rows == _sizes[0] && cols == _sizes[1] )
            return;
        for( i = 0; i < d; i++ )
            if( size[i] != _sizes[i] )
                break;
        if( i == d && (d > 1 || size[1] == 1))
            return;
    }

    release();
    if( d == 0 )
        return;
    flags = (_type & CV_MAT_TYPE_MASK) | MAGIC_VAL;
    setSize(*this, d, _sizes, 0, true);

    if( total() > 0 )
    {
#ifdef HAVE_TGPU
        if( !allocator || allocator == tegra::getAllocator() ) allocator = tegra::getAllocator(d, _sizes, _type);
#endif
        if( !allocator )
        {
            size_t totalsize = alignSize(step.p[0]*size.p[0], (int)sizeof(*refcount));
            data = datastart = (uchar*)fastMalloc(totalsize + (int)sizeof(*refcount));
            refcount = (int*)(data + totalsize);
            *refcount = 1;
        }
        else
        {
#ifdef HAVE_TGPU
           try
            {
                allocator->allocate(dims, size, _type, refcount, datastart, data, step.p);
                CV_Assert( step[dims-1] == (size_t)CV_ELEM_SIZE(flags) );
            }catch(...)
            {
                allocator = 0;
                size_t totalSize = alignSize(step.p[0]*size.p[0], (int)sizeof(*refcount));
                data = datastart = (uchar*)fastMalloc(totalSize + (int)sizeof(*refcount));
                refcount = (int*)(data + totalSize);
                *refcount = 1;
            }
#else
            allocator->allocate(dims, size, _type, refcount, datastart, data, step.p);
            CV_Assert( step[dims-1] == (size_t)CV_ELEM_SIZE(flags) );
#endif
        }
    }

    finalizeHdr(*this);
}
*/

}
