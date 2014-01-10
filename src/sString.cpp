/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

//
// SJIS string
//

#include "stdafx.h"
#include  <stdio.h>
#include  <stdarg.h>
#include  <string.h>

#include  "sString.h"
#ifdef USE_UNICODE_STRING
  #include  "uString.h"
#endif
#ifdef USE_JIS_STRING
  #include  "jString.h"
#endif
#include  "libjcode.h"

sjisString::sjisString( const char *str ) // "5%" などの文字列もそのまま取り込む。
{
    if( str == 0 )
    {
        ary[0]=0;
        return;
    }
    ary[(unsigned int)strlen(str)] = 0;   // extends
    strcpy( ary.rawData(), str );
}

/*
sjisString::sjisString( const char *format, ... )
{
    if( format == 0 )
    {
        ary[0]=0;
        return;
    }

    va_list arg_ptr;
    char buff[1024*8];
    va_start(arg_ptr,format);
    vsprintf(buff,format,arg_ptr);
    va_end(arg_ptr);

    ary[ strlen(buff) ] = 0;   // extends
    strcpy( ary.rawData(), buff );
}
*/

#ifdef USE_UNICODE_STRING
sjisString::sjisString( const unicodeString &str )
{
    int len = str.length() * 2;
    ary[len] = 0;  // extends
    if( wcstombs( ary.rawData(), str.pData(), len + 1 ) == -1 )
		ary[0] = 0;

}
#endif

#ifdef USE_JIS_STRING
sjisString::sjisString( const jisString &str )
{
    ary[ str.length() * 2 ] = 0;  // extends
    _seven2shift( (unsigned char *)str.pData(), (unsigned char *)ary.rawData() );
}
#endif

sjisString::~sjisString()
{
#ifdef _DEBUG
    if( ary.rawData() != 0 )
        ary[0] = 'X';  // for debug
#endif
}

// "char *" -> "sjisString" assignment.
sjisString & sjisString::operator = ( const char * s )
{
    if( s == 0 )
    {
        ary[0]=0;
        return *this;
    }

    ary[(unsigned int)strlen(s)] = 0;   // extends
    strcpy( ary.rawData(), s );
    return *this;
}

#ifdef USE_UNICODE_STRING
sjisString & sjisString::operator = ( const unicodeString & str )
{
    int len = str.length() * 2;
    ary[len] = 0;  // extends
    len = (int)wcstombs( ary.rawData(), str.pData(), len + 1 );
	if( len == -1 )
		ary[0] = 0;
    return *this;
}
#endif

#ifdef USE_JIS_STRING
sjisString & sjisString::operator = ( const jisString & str )
{
    ary[ str.length() * 2 ] = 0;  // extends
    _seven2shift( (unsigned char *)str.pData(), (unsigned char *)ary.rawData() );
    return *this;
}
#endif

sjisString & sjisString::operator += ( const sjisString &x )
{
    ary[ length() + x.length() ] = 0;   // extends
    strcat( ary.rawData(), x.pData() );
    return *this;
}

sjisString & sjisString::operator += ( char c )
{
    int len = length();
    ary[ len + 1 ] = 0;
    ary[ len ] = c;
    return *this;
}

int sjisString::length() const
{
    return (int)strlen( ary.pData() );
}

void sjisString::cutTrailingSpaces()
{
	int pos;
	for( pos = length() - 1; ary[pos] == ' '; --pos )
        ;
    ++pos;
    ary[pos] = 0;
}

void sjisString::toUpper()
{
    _strupr( ary.rawData() );
}

// substring operator
sjisString sjisString::operator () ( int pos ) const
{ 
    sjisString tmp; 
    if( pos < length() )
        tmp = ary.pData() + pos; 
    return tmp;
}

sjisString sjisString::operator () (int pos, int l) const
{
    sjisString tmp;
    if( pos < length() )
    {
        tmp = ary.pData() + pos;
        if( l < tmp.length() )
            tmp[l] = '\0';
    }
    return tmp;
}

// identity
int operator == ( sjisString const &x, sjisString const &y )
{ 
    return !strcmp( x.ary.pData(), y.ary.pData() );
}

int operator != ( sjisString const &x, sjisString const &y )
{ 
    return strcmp( x.ary.pData(), y.ary.pData() );
}

// 'printf'-like fomatter
sjisString format( const char *format, ... )
{
    sjisString retStr; 

    if( format == 0 || format[0] == 0 )
        return retStr;

    va_list arg_ptr;
    char buff[1024*8];
    va_start(arg_ptr,format);
    _vsnprintf( buff, 1024*8, format, arg_ptr );
    va_end(arg_ptr);

    retStr = buff;
    return retStr;
}

// addintion
const sjisString operator + (sjisString const &x, sjisString const &y)
{
    sjisString tmp(x); 
    tmp += y;
    return tmp; 
}

// hash number
unsigned int hashNumber( const sjisString &str )
{
    unsigned int val = 0;
    int len = str.length();
    int loopNum = len / 4;
    int rem = len % 4;
    int i;

    for( i = 0; i < loopNum; ++i )
    {
        val ^= (((int *)str.pData())[i]);
    }

    for( i = 0; i < rem; ++i )
    {
        val ^= str[loopNum*4 + i];
    }

    return val;
}

// ファイルパス str の拡張子を ext に変更したものを返す。
sjisString ChangeExtension( const sjisString &str_, const char *ext )
{
	sjisString str = str_;

	char *lastPosBackSlash = strrchr(str.rawData(), '\\');
	char *lastPosPeriod = strrchr(str.rawData(), '.');

	if((lastPosPeriod == NULL) || (lastPosBackSlash != NULL) && (lastPosPeriod < lastPosBackSlash))
	{
		// ファイル名は . を含まない。
		str += ext;
	}
	else
	{
		*(lastPosPeriod+1) = 0;
		str += ext;
	}

	return str;
}

// ファイルパス str のディレクトリー部分を dir に変更する。
sjisString ChangeDirPart( const sjisString &str, const char *dir_ )
{
	sjisString dir = dir_;
	if( dir.length() > 0 && dir[dir.length() - 1] != '\\')
		dir += '\\';

	return (dir + GetFileNameAndExtension(str));
}

// ファイルパス str の拡張子を得る。
sjisString GetExtension( const sjisString &str )
{
	sjisString ext;

	const char *lastPosBackSlash = strrchr(str.pData(), '\\');
	const char *lastPosPeriod = strrchr(str.pData(), '.');

	if((lastPosPeriod == NULL) || (lastPosBackSlash != NULL) && (lastPosPeriod < lastPosBackSlash))
	{
		// ファイル名は . を含まない。
	}
	else
	{
		ext = (lastPosPeriod + 1);
	}

	return ext;
}

// ファイルパス str のファイル名（拡張子を除く本体部分）に tail を追加する。
sjisString AppendToFileName( const sjisString &str_, const char *tail )
{
	sjisString str = str_;

	char *lastPosBackSlash = strrchr(str.rawData(), '\\');
	char *lastPosPeriod = strrchr(str.rawData(), '.');

	if((lastPosPeriod == NULL) || (lastPosBackSlash != NULL) && (lastPosPeriod < lastPosBackSlash))
	{
		// ファイル名は . を含まない。
		str += tail;
	}
	else
	{
		sjisString afterPeriod = lastPosPeriod;
		*lastPosPeriod = 0;
		str += tail;
		str += afterPeriod;
	}

	return str;
}


// ファイルパス str のファイル名（フォルダーや拡張子を除く本体部分）を返す。
sjisString GetFileName(const sjisString &str_)
{
	sjisString str = str_;

	char *lastPosBackSlash = strrchr(str.rawData(), '\\');

	if (lastPosBackSlash != NULL)
	{
		sjisString tmp = lastPosBackSlash + 1;
		str = tmp;
	}

	char *lastPosPeriod = strrchr(str.rawData(), '.');

	if (lastPosPeriod != NULL)
		*lastPosPeriod = 0;

	return str;
}

// ファイルパス str のファイル名（拡張子を含む）を返す。
sjisString GetFileNameAndExtension(const sjisString &str_)
{
	sjisString str = str_;

	char *lastPosBackSlash = strrchr(str.rawData(), '\\');

	if (lastPosBackSlash != NULL)
	{
		sjisString tmp = lastPosBackSlash + 1;
		return tmp;
	}

	return str;
}
