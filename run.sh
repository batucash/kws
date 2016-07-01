
##################### List of Input Vars################
outfile=../outputs
output=out.exe
####################Functions to Be Used ###################

###########################################################
g++ main.cpp -o $outfile/$output -O2 -I ~/Desktop/Research/software/armadillo-4.600.4/include -DARMA_DONT_USE_WRAPPER -lblas -llapack
#cd $outfile&
./$outfile/$output


