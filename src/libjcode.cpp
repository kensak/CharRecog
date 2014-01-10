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

#include "stdafx.h"
#define  LIBJCODE_CPP
#include "libjcode.h"

#define ASCII         0
#define JIS           1
#define EUC           2
#define SJIS          3
#define NEW           4
#define OLD           5
#define NEC           6
#define EUCORSJIS     7
#define UNKNOWN       9999
#define NUL           0
#define LF            10
#define FF            12
#define CR            13
#define ESC           27
#define SS2           142

#define KANJI         1
#define KANA          2

#define TRUE          1
#define FALSE         0

#define CHAROUT(ch) *str2 = (unsigned char)(ch); ++str2;

/* --------------------------------------- JIS(ISO-2022) コードへ切り替え -- */

unsigned char *_to_jis(unsigned char *str) {
  *str = (unsigned char)ESC; ++str;
  *str = (unsigned char)'$'; ++str;
  *str = (unsigned char)'B'; ++str;
  return str;
}

/* ----------------------------------------------- ASCII コードへ切り替え -- */

/* ESC ( B と ESC ( J の違い。
   本来は、 ESC ( J が正しいJIS-Roman 体系であるが、
   インターネットの上では、英数字はASCII の方が自然かと思われる。
   \ 記号と ~記号が違うだけである。 */

unsigned char *_to_ascii(unsigned char *str, char escapeOutChar) {
  *str = (unsigned char)ESC; ++str;
  *str = (unsigned char)'('; ++str;
  // // 2005年7月15日　コニカミノルタにて "ESC ( B" に修正。
  // //*str = (unsigned char)'B'; ++str;
  // *str = (unsigned char)'J'; ++str;
  *str = (unsigned char)escapeOutChar; ++str;
  return str;
}

/* ----------------------------------------------- 半角カナ・コードへ切り替え -- */
unsigned char *_to_kana(unsigned char *str, char escapeKanaChar) {
  *str = (unsigned char)ESC; ++str;
  *str = (unsigned char)')'; ++str;
  // //*str = (unsigned char)'B'; ++str;
  // *str = (unsigned char)'I'; ++str;
  *str = (unsigned char)escapeKanaChar; ++str;
  return str;
}

/* -------------------------------------- JIS コード を SJISとしてシフト -- */

//void _jis_shift(int *p1, int *p2)
void _jis_shift(unsigned short *p1, unsigned short *p2)
{
#define hic (*p1)
#define loc (*p2)

    const int plane = 1;

    /* JIS X 0213:2000 */
    if (plane == 2) {
        if      (hic == 0x21 || (0x23 <= hic && hic <= 0x25))
            hic += 0x5e;
        else if (hic == 0x28 || (0x2c <= hic && hic <= 0x2f))
            hic += 0x58;
        else if (hic >= 0x6e)
            hic += 0x1a;
        else
        {
            hic = loc = 0;
            return;
        }
    }
    /* jis -> sjis */
    loc += (unsigned short)( (hic & 0x01) ? 0x1f : 0x7d );
    if (loc >= 0x7f) ++loc;
    hic = (unsigned short)( ((hic - 0x21) >> 1) + 0x81 );
    if (hic > 0x9f) hic += 0x40;

    /*    
    // 昔のコード
    unsigned char c1 = *p1;
    unsigned char c2 = *p2;
    int rowOffset = c1 < 95 ? 112 : 176;
    int cellOffset = c1 % 2 ? (c2 > 95 ? 32 : 31) : 126;
    
    *p1 = ((c1 + 1) >> 1) + rowOffset;
    *p2 += cellOffset;
    */
}

/* --------------------------------- SJIS コードをJIS コードとしてシフト -- */

//void _sjis_shift(int *p1, int *p2)
void _sjis_shift(unsigned short *p1, unsigned short *p2)
{
    int plane;

    static unsigned short table[][2] = {
        {0x81, 0x84}, {0x82, 0x82}, {0x83, 0x86},
        {0x87, 0x87}, {0x88, 0xe7}
    };

    /* JIS X 0213:2000 */
    plane = (0xf0 <= hic) ? 2 : 1;
    if (0xf0 <= hic && hic <= 0xf4)
        hic = table[hic - 0xf0][(loc <= 0x9e) ? 0 : 1];
    else if (hic >= 0xf5) hic -= 0xd;
    /* sjis -> jis */
    hic -= (unsigned short) (hic <= 0x9f) ? 0x71 : 0xb1;
    hic  = (unsigned short) (hic << 1) + 1;
    if (loc > 0x9e) ++hic;
    if (loc > 0x7f) --loc;
    loc -= (unsigned short) (loc >= 0x9e) ? 0x7d : 0x1f;

    if( plane == 2 )
    {
        // 第4水準漢字は '〓' で置き換える。
        hic = (unsigned short) libjcodeFlags.jisCodeForLevel4kanjiSubstitute >> 8;
        loc = (unsigned short) libjcodeFlags.jisCodeForLevel4kanjiSubstitute & 0xff;
    }

    /*
    unsigned char c1 = *p1;
    unsigned char c2 = *p2;
    int adjust = c2 < 159;
    int rowOffset = c1 < 160 ? 112 : 176;
    int cellOffset = adjust ? (c2 > 127 ? 32 : 31) : 126;

    *p1 = ((c1 - rowOffset) << 1) - adjust;
    *p2 -= cellOffset;
    */
}

#define HANKATA(a)  (a >= 161 && a <= 223)
/* ---------------------------------------------- SJIS 半角を全角に変換 -- */
/*
#define ISMARU(a)   (a >= 202 && a <= 206)
#define ISNIGORI(a) ((a >= 182 && a <= 196) || (a >= 202 && a <= 206) || (a == 179))

static int stable[][2] = {
    {129,66},{129,117},{129,118},{129,65},{129,69},{131,146},{131,64},
    {131,66},{131,68},{131,70},{131,72},{131,131},{131,133},{131,135},
    {131,98},{129,91},{131,65},{131,67},{131,69},{131,71},{131,73},
    {131,74},{131,76},{131,78},{131,80},{131,82},{131,84},{131,86},
    {131,88},{131,90},{131,92},{131,94},{131,96},{131,99},{131,101},
    {131,103},{131,105},{131,106},{131,107},{131,108},{131,109},
    {131,110},{131,113},{131,116},{131,119},{131,122},{131,125},
    {131,126},{131,128},{131,129},{131,130},{131,132},{131,134},
    {131,136},{131,137},{131,138},{131,139},{131,140},{131,141},
    {131,143},{131,147},{129,74},{129,75}};

unsigned char *_sjis_han2zen(unsigned char *str, int *p1, int *p2)
{
  register int c1, c2;

  c1 = (int)*str; ++str;
  *p1 = stable[c1 - 161][0];
  *p2 = stable[c1 - 161][1];

  // 濁音、半濁音の処理
  c2 = (int)*str;
  if (c2 == 222 && ISNIGORI(c1)) {
    if ((*p2 >= 74 && *p2 <= 103) || (*p2 >= 110 && *p2 <= 122))
      (*p2)++;
    else if (*p1 == 131 && *p2 == 69)
      *p2 = 148;
    ++str;
  }

  if (c2 == 223 && ISMARU(c1) && (*p2 >= 110 && *p2 <= 122) ) {
    *p2 += 2;
    ++str;
  }
  return str++;  // 正しい？
}
*/

/* -------------------------------------------------- SJIS を JIS に変換 -- */

// #define SJIS1(A)    ((A >= 0x81 && A <= 0x9f) || (A >= 0xed && A <= 0xef))
#define SJIS1(A)    ((A >= 0x81 && A <= 0x9f) || (A >= 0xe0 && A <= 0xfc))
#define SJIS2(A)    (A >= 0x40 && A <= 0xfc)

void _shift2seven(unsigned char *str, unsigned char *str2, char escapeOutChar, char escapeKanaChar)
{
    // int p1,p2,esc_in = FALSE;
    unsigned short p1,p2,esc_in = FALSE;
    
    // while ((p1 = (int)*str) != '\0') {
    while ((p1 = (unsigned short)*str) != '\0') {
        if (SJIS1(p1)) {
            //if((p2 = (int)*(++str)) == '\0') break;
            if((p2 = (unsigned short)*(++str)) == '\0') break;
            if (SJIS2(p2)) {
                _sjis_shift(&p1,&p2);
                if (esc_in != KANJI) {
                    esc_in = KANJI;
                    str2 = _to_jis(str2);
                }
            }
            CHAROUT(p1);
            CHAROUT(p2);
            ++str;
            continue;
        }
        
		if( escapeKanaChar != '\0' )
		{
			// 半角カナをエスケープする。
			if (HANKATA(p1)) {
				if (esc_in != KANA) {
					esc_in = KANA;
					str2 = _to_kana(str2, escapeKanaChar);
				}
				CHAROUT(p1);
				++str;
				continue;
			}
		}
        
        if (esc_in) {
            // LF / CR の場合は、正常にエスケープアウトされる
            esc_in = ASCII;
            str2 = _to_ascii(str2, escapeOutChar);
        }
        CHAROUT(p1);
        ++str;
    }
    
    if (esc_in)
        str2 = _to_ascii(str2, escapeOutChar);
    *str2='\0';
}


/* ------------------------------------------------- 半角 SJIS を取り除く -- */
/*
void _shift_self(unsigned char *str, unsigned char *str2)
{
  int p1;
  
  while ((p1 = (int)*str) != '\0') {
    CHAROUT(p1);
    ++str;
  }
  *str2='\0';
}
*/

/* -------------------------------------- ESC シーケンスをスキップする ----- */

unsigned char *_skip_esc(unsigned char *str, unsigned short *esc_in) {
  int c;
  
  c = (int)*(++str);
  if ((c == '$') || (c == '(') || (c == ')')) ++str;
  if ((c == 'K') || (c == '$'))
      *esc_in = TRUE;
  else
      *esc_in = FALSE;

  if(*str != '\0') ++str;
  return str;
}

/* ----------------------------------------------- JIS を SJIS に変換する -- */

void _seven2shift(unsigned char *str, unsigned char *str2)
{
    //int p1, p2, esc_in = FALSE;
    unsigned short p1, p2, esc_in = FALSE;

    // while ((p1 = (int)*str) != '\0') {
    while ((p1 = (unsigned short)*str) != '\0') {
        
        // ESCシーケンスをスキップする
        if (p1 == ESC) {
            str = _skip_esc(str, &esc_in);
            continue;
        }
        
        if (p1 == LF || p1 == CR) {
            if (esc_in) esc_in = FALSE;
        }
        
        if(esc_in) { // ISO-2022-JP コード
            // if((p2 = (int)*(++str)) == '\0') break;
            if((p2 = (unsigned short)*(++str)) == '\0') break;
            
            _jis_shift(&p1, &p2);
            
            CHAROUT(p1);
            CHAROUT(p2);
        }else{       // ASCII コード
            CHAROUT(p1);
        }
        ++str;
    }
    *str2 = '\0';
}








