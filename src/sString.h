/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

//
// SJIS string
//

#ifndef __SSTRING_H__
#define __SSTRING_H__

#include  "aString.h"

class unicodeString;
class jisString;

class sjisString : public aString<char>
{

public:

    sjisString( int len = 0 ) : aString<char>( len ) {}
    sjisString( const char *str );  // "5%" などの文字列もそのまま取り込む。
    // explicit sjisString( const char *format, ... );
    explicit sjisString( const unicodeString &str );
    explicit sjisString( const jisString &str );
    ~sjisString();

    sjisString & operator = ( const char * const s );
    sjisString & operator = ( const jisString & str );
    sjisString & operator = ( const unicodeString & str );

    sjisString & operator += ( const sjisString &x );
    sjisString & operator += ( char c );

    int length() const;

    void cutTrailingSpaces();
    void toUpper();

    // substring operator
    sjisString operator () ( int pos ) const;
    sjisString operator () ( int pos, int l ) const;

    // friend functions: identity
    friend int operator == ( sjisString const &x, sjisString const &y );
    friend int operator != ( sjisString const &x, sjisString const &y );
};

// addintion
const sjisString operator + ( sjisString const &x, sjisString const &y );

// 'printf'-type fomatter
sjisString format( const char *format, ... );

// hash number
unsigned int hashNumber( const sjisString &str );

// ファイルパス str の拡張子を ext に変更する。
sjisString ChangeExtension( const sjisString &str, const char *ext );

// ファイルパス str のディレクトリー部分を dir に変更する。
sjisString ChangeDirPart( const sjisString &str, const char *dir );

// ファイルパス str の拡張子を得る。
sjisString GetExtension( const sjisString &str );

// ファイルパス str のファイル名（拡張子を除く本体部分）に tail を追加する。
sjisString AppendToFileName( const sjisString &str, const char *tail );

// ファイルパス str のファイル名（拡張子を除く本体部分）を返す。
sjisString GetFileName(const sjisString &str);

// ファイルパス str のファイル名（拡張子を含む）を返す。
sjisString GetFileNameAndExtension(const sjisString &str);

#endif

