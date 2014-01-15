CharRecog : How To Build
=====================

Here are the explanation of my building environment and
some instructions about how to build `CharRecog.exe`.

Environment
-----------
OS : Windows 7, 64 bit version  
RAM : 8GB

Requirements
------------
+ Micosoft Visual Studio Express 2012 for Windows Desktop  
  [You can download it from here.][MS]

+ OpenCV 2.4.3  
  [You can download it from here.][OpenCV]  
  Build the library according to the instructions.
  You may need some additional softwares in course of building process.  
  (In the included CharRecog project, OpenCV is statically linked,
  whereas the C/C++ runtime is dynamically linked.)
  
+ C++ AMP BLAS Library 1.0  
  [You can download it from here.][ampblas]
  
Building Steps
--------------
1. Open `src/CharRecog.vcxproj` in Visual Studio.
1. Open the property page of the project, and adjust the following settings according to the actual environment.
 - Configuration Properties -> VC++ directories -> include directories
 - Configuration Properties -> VC++ directories -> library directories
1. Choose the correct configuration (Release/Debug) and platform (win32/x64), and build the project.

Tips
----
+ If you prefer doing all the calculations in 'double', undefine `REAL_IS_FLOAT` defined in `config.h`.
+ Log messages will be written in `log/log.txt` by default.
  To stop it, undefine `DEBUG_OUTPUT_LOG` defined in `config.h`.
  
Ken Sakakibara  
Email: ken.sakakibar@gmail.com  
Blog: [http://kensak.github.io/](http://kensak.github.io/)  
GitHub: [https://github.com/kensak](https://github.com/kensak)  
Twitter: KenSakakibar

 
[MS]: http://msdn.microsoft.com/en-us/library/vstudio/dd831853(v=vs.110).aspx
[OpenCV]: http://sourceforge.net/projects/opencvlibrary/files/opencv-win/
[ampblas]: http://ampblas.codeplex.com/releases/view/92383

