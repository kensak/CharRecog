/* 
   Copyright (c) Ken Sakakibara 2014.

   Dual licensed under the MIT license http://www.opensource.org/licenses/mit-license.php
   and GPL license http://www.gnu.org/licenses/gpl.html .

*/

#include "stdafx.h"
#include "sArray.h"

static void defaultLogErrorFunc( char *errorStr ) { printf(errorStr); }

void (*sArrayLogErrorFunc)(char *errorStr) = defaultLogErrorFunc;

void sArraySetLogErrorFunc( void (*_logErrorFunc)(char *errorStr) )
{
    sArrayLogErrorFunc = _logErrorFunc;
}

