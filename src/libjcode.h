/* 
 *  libjcode.c -- 漢字変換ライブラリ    1.0 版
 *                (C) Kuramitsu Kimio, Tokyo Univ. 1996-97
 *
 *  このライブラリは、CGI Programming with C and Perl のために
 *  Ken Lunde 著 「日本語情報処理」 (O'llery) を参考にして、
 *  ストリーム用だったjconv.c を、ストリング対応にしてライブラリ化
 *  しました。 
 *  ただし、CGI (INTERNET)での利用を考えて、変更してあります。
 */

// Modified by Ken Sakakibara.

#ifndef __LIBJCODE_H__
#define __LIBJCODE_H__

struct libjcodeFlagsType
{
    libjcodeFlagsType() : jisCodeForLevel4kanjiSubstitute( 0x222e ) {} // '〓'

    unsigned short jisCodeForLevel4kanjiSubstitute;
};

#ifdef LIBJCODE_CPP
    #define EXTERN
#else
    #define EXTERN extern
#endif

EXTERN libjcodeFlagsType libjcodeFlags;

// SJIS -> JIS
EXTERN void _shift2seven(unsigned char *str, unsigned char *str2, char escapeOutChar = 'J', char escapeKanaChar = '\0');

// JIS -> SHIS
EXTERN void _seven2shift(unsigned char *str, unsigned char *str2);

#endif

