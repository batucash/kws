
##################### List of Input Vars################
outfile=../outputs
output=out.exe
####################Functions to Be Used ###################

###########################################################
g++ main.cpp -o $outfile/$output -O2 -I ~/Desktop/Research/software/armadillo-7.200.2/include -DARMA_DONT_USE_WRAPPER -lblas -llapack
#cd $outfile&
./$outfile/$output


