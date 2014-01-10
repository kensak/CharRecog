/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

#include "StdAfx.h"
#include "sArray.h"
#include "log.h"

#ifdef DEBUG_OUTPUT_LOG
FILE *fpLog = NULL;
#endif

#ifdef DEBUG_OUTPUT_LOG
void LogA( const char *format, ... )
{
    if (fpLog == NULL || format == 0 || format[0] == 0)
        return;

    va_list arg_ptr;
    char buff[1024*8];
    va_start( arg_ptr, format );
    _vsnprintf_s( buff, sizeof(buff), _TRUNCATE, format, arg_ptr );
    va_end( arg_ptr );

    //fputs(buff, stdout);
    fputs(buff, fpLog);
	fflush(fpLog);
}

void LogW( const wchar_t *format, ... )
{
    if (fpLog == NULL || format == 0 || format[0] == _T('\0'))
        return;

    va_list arg_ptr;
    wchar_t buff[1024*8];
    va_start( arg_ptr, format );
    _vsnwprintf_s( buff, sizeof(buff)/sizeof(wchar_t), _TRUNCATE, format, arg_ptr );
    va_end( arg_ptr );

	//fputws(buff, stdout);

	//
	// fputws(buff, fpLog); ���� _istprint() �� OK �Ȃ̂ɕ\���ł��Ȃ�����������Ƃ��ɁA�����\������Ȃ��Ȃ��Ă��܂��B
	// putwc() ���g���Ε\���ł��镶�������͕\�������B
	// �Ȃ� _istprint() �� OK �Ȃ̂� fputws() �� putwc() �ŕ\���ł��Ȃ����͓�B
	//
	size_t len = wcslen(buff);
	for(unsigned int i = 0; i < len; ++i)
		putwc(buff[i], fpLog);

	fflush(fpLog);
}

#endif

// log function for JPEG library
static void logErrorFunc( char *errorStr )
{
    LogA( errorStr );
}

void InitializeLog()
{
  const char *funcName = "InitializeLog";

#ifdef DEBUG_OUTPUT_LOG

  const char *logDir = "log";
  const char *logPath = "log\\log.txt";

  if (_access_s(logDir, 0) != 0 && _mkdir(logDir) != 0)
  {
    printf_s("%s: ���O�E�t�H���_�[ %s �����܂���B\n", funcName, logDir);
    return;
  }

	errno_t errnum = fopen_s(&fpLog, logPath, "at");
  if (errnum != 0)
  {
    char buffer[256];
    strerror_s(buffer, errnum);
    printf_s("%s: �t�@�C�� %s ���J���܂���B%s\n", funcName, logPath, buffer);
    return;
  }
#endif

	sArraySetLogErrorFunc( logErrorFunc );
}

void FinalizeLog()
{
#ifdef DEBUG_OUTPUT_LOG
	if (fpLog != NULL)
  {
		fclose( fpLog );
    fpLog = NULL;
  }
#endif
}