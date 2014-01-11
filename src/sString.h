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
    sjisString( const char *str );  // "5%" �Ȃǂ̕���������̂܂܎�荞�ށB
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

// �t�@�C���p�X str �̊g���q�� ext �ɕύX����B
sjisString ChangeExtension( const sjisString &str, const char *ext );

// �t�@�C���p�X str �̃f�B���N�g���[������ dir �ɕύX����B
sjisString ChangeDirPart( const sjisString &str, const char *dir );

// �t�@�C���p�X str �̊g���q�𓾂�B
sjisString GetExtension( const sjisString &str );

// �t�@�C���p�X str �̃t�@�C�����i�g���q�������{�̕����j�� tail ��ǉ�����B
sjisString AppendToFileName( const sjisString &str, const char *tail );

// �t�@�C���p�X str �̃t�@�C�����i�g���q�������{�̕����j��Ԃ��B
sjisString GetFileName(const sjisString &str);

// �t�@�C���p�X str �̃t�@�C�����i�g���q���܂ށj��Ԃ��B
sjisString GetFileNameAndExtension(const sjisString &str);

#endif

