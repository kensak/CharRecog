/* 
 *  libjcode.c -- �����ϊ����C�u����    1.0 ��
 *                (C) Kuramitsu Kimio, Tokyo Univ. 1996-97
 *
 *  ���̃��C�u�����́ACGI Programming with C and Perl �̂��߂�
 *  Ken Lunde �� �u���{���񏈗��v (O'llery) ���Q�l�ɂ��āA
 *  �X�g���[���p������jconv.c ���A�X�g�����O�Ή��ɂ��ă��C�u������
 *  ���܂����B 
 *  �������ACGI (INTERNET)�ł̗��p���l���āA�ύX���Ă���܂��B
 */

// Modified by Ken Sakakibara.

#ifndef __LIBJCODE_H__
#define __LIBJCODE_H__

struct libjcodeFlagsType
{
    libjcodeFlagsType() : jisCodeForLevel4kanjiSubstitute( 0x222e ) {} // '��'

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

