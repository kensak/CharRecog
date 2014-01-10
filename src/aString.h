/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

//
// abstract string
//

#ifndef __ASTRING_H__
#define __ASTRING_H__

#include  "sArray.h"

// 4710 : XXXXX ÇÕ≤›◊≤›ä÷êîÇ≈ÇÕÇ†ÇËÇ‹ÇπÇÒ
#pragma warning( disable : 4710 )

template<class T>
class aString
{

public:

    aString( int len = 0 );
    aString( const aString &x );
    aString( T c, int len = 1 );
    virtual ~aString() {}

    aString & operator = ( const aString &x );

    // "aString" -> "T *" convertors (access to pointer)
    //--- reference only
    const T * pData() const;
    //--- allow modification
    T * rawData();

    // substring operator
    //aString operator () ( int pos ) const;
    //aString operator () ( int pos, int l ) const;

    // indirection
    //--- reference only
    T   operator[] ( int idx ) const;
    //--- allow modification
    T & operator[] ( int idx );

protected:
    
    sArray<T> ary;
};

//---------------- implement -----------------

template<class T>
aString<T>::aString( int len ) : ary( len + 1 )
{
    ary[0] = T( 0 );
}

template<class T>
aString<T>::aString( const aString &x )  : ary( x.ary )
{
}

template<class T>
aString<T>::aString( T c, int len ) : ary( len + 1 )
{
    ary.clear( len, c );
    ary[len] = T( 0 );
}

template<class T>
aString<T> & aString<T>::operator = ( const aString<T> &x )
{
    if( this != &x )
    {
        //len = x.len;
        ary = x.ary;
    }
    return *this;
}

// "aString" -> "T *" convertors (access to pointer)
template<class T>
const T * aString<T>::pData() const
{ 
    return ary.pData();
}

template<class T>
T * aString<T>::rawData()
{ 
    return ary.rawData();
}

// substring operator
//aString operator () ( int pos ) const;
//aString operator () ( int pos, int l ) const;

template<class T>
T aString<T>::operator[] ( int idx ) const
{
    return ary[idx];
}

template<class T>
T & aString<T>::operator[] ( int idx )
{
    return ary[idx];
}

#pragma warning( default : 4710 )

#endif

