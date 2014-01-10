/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

//
// class fArray: finite array with a length
//

#ifndef __FARRAY_H__
#define __FARRAY_H__

#include  <assert.h>
#include  "sArray.h"

// 4101     'tmp' : ﾛｰｶﾙ変数は 1 度も使われません。
// 4189     'tmp' : ﾛｰｶﾙ変数が初期化されましたが、参照されていません
#pragma warning( disable : 4101 4189 )

template<class T>
class fArray
{

public:

    // constructors
    fArray( unsigned int _len = 0, unsigned int allocated = 0 ) : ary( allocated ), len( _len )
    {
        if( _len > allocated )
        {
            T tmp = ary[_len-1];
        }
    }

    fArray( const fArray<T> &fa ) : ary( fa.ary ), len( fa.len ) {}

    // destructor
    ~fArray() {}

    fArray & operator = ( const fArray &fa );

	// データを fa へ上書きし、this は長さゼロにする。
    fArray & moveTo( fArray &fa );

    // indirection which does not resize.

    //--- allow modification
    T & operator[] (unsigned int idx);
    //--- reference only 
    T const & operator[] (unsigned int idx) const;


    // indirection which may resize
    // (when the given index is greater than the length of the array)

    //--- allow modification
    T & operator() (unsigned int idx);
    //--- reference only 
    T const & operator() (unsigned int idx) const;


    // access to pointer

    //--- reference only 
    const T * pData() const;
    //--- allow modification
    T * rawData();


    void append( const T& x );
    void append(       T& x );
	void concat( const fArray<T> &fa );
    void push_back( const T& x ) { append( x ); }
    T pop_last();
    // x と同じものを先頭から探し、見つかったらそれをアレイの最後の要素で上書きし、アレイの長さを１減らす。
    // 見つかったら true を、そうでなければ false を返す。
    bool del( const T& x );
	void delAt( unsigned int index );
	void insertAt( unsigned int index, const T& x );
    T first() const { return ary[ 0 ]; }
    T last() const { return ary[ len - 1 ]; }

    unsigned int length() const { return len; }
    void setLength( unsigned int _len );
    void expand( int diff )
    {
        if( diff >= 0 )
            setLength( len + (unsigned int)diff );
        else
            setLength( len - (unsigned int)(-diff) );
    }
    void clear( const T& init );  // アレイを init で埋める。
    void freeMem();  // メモリーを解放する。長さは 0 になる。

    void sort( int (__cdecl *compare)(const void *elem1, const void *elem2) );

    // friend functions: identity
    friend int operator == ( fArray<T> const &x, fArray<T> const &y );
    friend int operator != ( fArray<T> const &x, fArray<T> const &y );

private:

    sArray<T> ary;
    unsigned int len;

};

template<class T>
fArray<T> & fArray<T>::operator = ( const fArray<T> &fa )
{ 
    if( this != &fa )
    {
        ary = fa.ary;
        len = fa.len;
    }
    return *this;
}

template<class T>
fArray<T> & fArray<T>::moveTo( fArray<T> &fa )
{
    if( this != &fa )
    {
		ary.moveTo( fa.ary );
		fa.len = len;
		len = 0;
	}
    return fa;
}

// indirection which does not resize.

//--- allow modification
template<class T>
T & fArray<T>::operator[] (unsigned int idx)
{
    if( len <= idx )
        assert( 0 );
    return ary[idx];
}

//--- reference only 
template<class T>
T const & fArray<T>::operator[] (unsigned int idx) const
{
    if( len <= idx )
        assert( 0 );
    return ary[idx];
}

// indirection which may resize
// (when the given index is greater than the length of the array).

//--- allow modification
template<class T>
T & fArray<T>::operator() (unsigned int idx)
{
    if( len <= idx )
        len = idx + 1;
    return ary[idx];
}

//--- reference only 
template<class T>
T const & fArray<T>::operator() (unsigned int idx) const
{
    if( len <= idx )
        len = idx + 1;
    return ary[idx];
}

// access to pointer

//--- reference only 
template<class T>
const T * fArray<T>::pData() const
{
    return ary.pData();
}

//--- allow modification
template<class T>
T * fArray<T>::rawData()
{
    return ary.rawData(); 
}

template<class T>
void fArray<T>::append( const T& x )
{
    ++len;
    ary[len-1] = x;
}

template<class T>
void fArray<T>::append( T& x )
{
    ++len;
    ary[len-1] = x;
}

template<class T>
void fArray<T>::concat( const fArray<T> &fa )
{
	unsigned int srcLen = fa.length();
    for( unsigned int i = 0; i < srcLen; ++i )
    {
		append( fa[i] );
	}
}

template<class T>
T fArray<T>::pop_last()
{
    if( len )
    {
        --len;
        return ary[len];
    }
    assert( 0 );
    return T();
}

template<class T>
bool fArray<T>::del( const T& x )
{
    for( unsigned int i = 0; i < len; ++i )
    {
        if( ary[i] == x )
        {
            if( i != len - 1 )
                ary[i] = ary[len-1];
            --len;
            return true;
        }
    }
    return false;
}

template<class T>
void fArray<T>::delAt( unsigned int index )
{
	assert( index < len );

    for( unsigned int i = index; i < len-1; ++i )
    {
		ary[i] = ary[i+1];
	}
    --len;
}

template<class T>
void fArray<T>::insertAt( unsigned int index, const T& x )
{
	T tmp = ary[len];  // resize

    for( int i = len-1; i >= (int)index; --i )
    {
		ary[i+1] = ary[i];
	}
    ary[index] = x;
    ++len;
}

template<class T>
void fArray<T>::setLength( unsigned int _len )
{
    assert( _len >= 0 );
    
    if( _len > len )
    {
        T tmp = ary[_len-1];  // resize
    }
    len = _len;
}

template<class T>
void fArray<T>::clear( const T& init )
{
    for( unsigned int i = 0; i < len; ++i )
        ary[i] = init;
}

template<class T>
void fArray<T>::freeMem()
{
    ary.freeMem();
    len = 0;
}

template<class T>
void fArray<T>::sort( int (__cdecl *compare)(const void *elem1, const void *elem2) )
{
    qsort( ary.rawData(), len, sizeof(T), compare );
}

// friend functions ========================================

// identity
template<class T>
int operator == ( fArray<T> const &x, fArray<T> const &y )
{
	if( x.length() != y.length() )
		return false;

	int len = x.length();
	for( int i = 0; i < len; ++i )
	{
		if( x[i] != y[i] )
			return false;
	}

    return true;
}

template<class T>
int operator != ( fArray<T> const &x, fArray<T> const &y )
{ 
    return !( x == y );
}

#pragma warning( default : 4101 4189 )

#endif

