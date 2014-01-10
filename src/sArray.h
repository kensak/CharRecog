/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

//
// class sArray: simple array of infinite length
//

#ifndef __SARRAY_H__
#define __SARRAY_H__

#include  <assert.h>

// エラーを表示するための関数を登録する関数
// 
// エラーが起きると登録された logging function が呼び出され、そのあとすぐにリターンする。
// このときエラーが起きた関数は特別なリターン・コードを返さないので、その後の処理を変えたい場合は logging function の
// 中から throw しなければならない。
//
// この関数で登録しないと、渡された文字列をただ printf で表示する関数がエラーの表示に使われます。
extern void sArraySetLogErrorFunc( void (*logErrorFunc)(char *errorStr) );

template<class T>
class sArray
{
    
public:
    
    sArray( unsigned int _size = 0 );
    sArray( const sArray<T> &ar );
    
    ~sArray() { delete [] ary; }
    
    sArray<T> & operator = (const sArray<T> &ar);
    
	// データを ar へ上書きし、this は長さゼロにする。
    sArray & moveTo( sArray &ar );

    // indirection which may resize
    // (when the given index is greater than the length of the array)
    //--- reference only 
    const T & operator[] (unsigned int idx) const;
    //--- allow modification
    T & operator[] (unsigned int idx);
    
    // access to pointer
    //--- reference only 
    const T * pData() const;
    //--- allow modification
    T * rawData();
    

    // sort elements whose index is in [head, tail].
    //    inline void sort(int head, int tail, int (*compare)(T *, T *))
    //        { qsort((void *)&(ary[head]), tail-head+1, sizeof(T), compare); }
    
    void clear( unsigned int len, T initVal );
    void freeMem();

private:
    
    T *ary;
    unsigned int size;
    
    void extend( unsigned int idx );
};


//=======================================================================================
//
// implement
//
//=======================================================================================

extern void (*sArrayLogErrorFunc)(char *errorStr);

template<class T>
void sArray<T>::extend( unsigned int idx )
{
    if( idx < size ) return;
    
    unsigned int new_size = size, i;
    T *new_ary;
    
    if( new_size == 0 )
        new_size = 1;
	// msb は most significant bit だけが 1 の unsigned int.
	static const unsigned int msb = 1u << (sizeof(unsigned int) * 8 - 1);
    while( new_size < msb && new_size <= idx )
        new_size *= 2;
	if(new_size <= idx)
	{
		// ここに来るのはたぶんバグ。
        assert( 0 );
        (*sArrayLogErrorFunc)("sArray: too much memory requested. Check out if this is not a bug.\n");
		new_size = UINT_MAX;
	}

    new_ary = new T[new_size];
    
    if( new_ary == 0 )
    {
        (*sArrayLogErrorFunc)("sArray: out of memory in extend().\n");
        return;
    }

    for( i = 0; i < size; ++i ) 
        new_ary[i] = ary[i];
    size = new_size;
    if( ary != 0 )
        delete [] ary;
    ary = new_ary;
}

    
// constructors.
template<class T>
sArray<T>::sArray( unsigned int _size ) : size( _size )
{
    if( size == 0 )
        ary = 0;
    else
    {
        ary = new T[size];
        if( ary == 0 )
        {
            (*sArrayLogErrorFunc)("sArray: out of memory in constructor 1.\n");
            return;
        }
    }
}
    
template<class T>
sArray<T>::sArray( const sArray<T> &ar ) : size( ar.size )
{
    if( size == 0 )
        ary = 0;
    else
    {
        ary = new T[size];
        if( ary == 0 )
        {
            (*sArrayLogErrorFunc)("sArray: out of memory in constructor 2.\n");
            return;
        }
    }
    for( unsigned int i = 0; i < size; ++i ) 
        ary[i] = ar.ary[i];
}

// member functions.
template<class T>
sArray<T> & sArray<T>::operator = (const sArray<T> &ar)
{
    if( this != &ar ) 
    {
        unsigned int i;
        if( size < ar.size ) 
        {
            if( ary != 0 )
                delete [] ary;
            size = ar.size;
            
            ary = new T[size];
            
            if( ary == 0 )
            {
                (*sArrayLogErrorFunc)("sArray: out of memory in operator = .\n");
		        assert( 0 );
                return *this;
            }

            for( i = 0; i < size; ++i )
                ary[i] = ar.ary[i];
        }
        else 
        {
            for( i = 0; i < ar.size; ++i )
                ary[i] = ar.ary[i];
        }
    }
    return *this;
}

template<class T>
sArray<T> & sArray<T>::moveTo( sArray<T> &ar )
{
    if( this != &ar )
    {
		ar.freeMem();
		ar.ary = ary;     // ポインタを付け替える。
		ar.size = size;

		// 自分は空にする。
		ary = 0;
		size = 0;
	}
    return ar;
}

// indirection which may resize
// (when the given index is greater than the length of the array).
//--- reference only 
template<class T>
const T & sArray<T>::operator[] ( unsigned int idx ) const
{
    if( size <= idx )
    {
        sArray<T> * const tmp = (sArray<T> * const)this;  // remove constness.
        tmp->extend(idx);
    }
    return ary[idx];
}

//--- allow modification
template<class T>
T & sArray<T>::operator[] ( unsigned int idx )
{ 
    if( size <= idx ) 
        extend(idx);
    return ary[idx]; 
}

// access to pointer

//--- reference only 
template<class T>
const T * sArray<T>::pData() const
{
    return ary;
}

//--- allow modification
template<class T>
T * sArray<T>::rawData()
{
    return ary; 
}

// sort elements whose index is in [head, tail].
//    inline void sort(int head, int tail, int (*compare)(T *, T *))
//        { qsort((void *)&(ary[head]), tail-head+1, sizeof(T), compare); }

template<class T>
void sArray<T>::clear( unsigned int len, T initVal )
{
    extend( len - 1 );
    for( unsigned int i = 0; i < len; ++i ) 
        ary[i] = initVal;
}

template<class T>
void sArray<T>::freeMem()
{
    if( ary != 0 )
        delete [] ary;
    ary = 0;
    size = 0;
}

#endif

