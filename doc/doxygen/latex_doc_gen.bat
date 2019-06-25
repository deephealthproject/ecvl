copy Doxyfile Doxyfile_LaTeX
echo GENERATE_LATEX=YES >> Doxyfile_LaTeX
doxygen Doxyfile_LaTeX
copy doxygen.sty ..\latex\doxygen.sty /Y
call ..\latex\make.bat
copy ..\latex\refman.pdf .\refman.pdf /Y
del /f Doxyfile_LaTeX
del /f refman_final.pdf
call pdftk refman.pdf cat 3-end output refman_final.pdf
del /f refman.pdf