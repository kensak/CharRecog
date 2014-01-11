/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL v2 license http://www.gnu.org/licenses/gpl.html .

*/

#pragma once

#include "config.h"

#ifdef DEBUG_OUTPUT_LOG
 extern void LogA( const char *format, ... );
 extern void LogW( const wchar_t *format, ... );
#else
 #define LogA  printf
 #define LogW wprintf
#endif

#ifdef  UNICODE
 #define Log  LogW
#else
 #define Log  LogA
#endif

void InitializeLog();
void FinalizeLog();
