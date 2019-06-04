doxygen Doxyfile_LaTeX
copy doxygen.sty ..\latex\doxygen.sty /Y
call ..\latex\make.bat
copy ..\latex\refman.pdf .\refman.pdf /Y