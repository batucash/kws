

#include <armadillo>
#include <vector>
#include <string>

using namespace std;
using namespace arma;
const uword NUMPHONES = 43;
float mu = 0.002;
float mu2 = 0.0002;
uword HIDDEN = 43;
float sigmoid (float x);
struct training {
    uword x;
    uword y;
    uword r;
};

int main(int argc, char *argv[]){
    
    //  1.  Read frames and labels
    
    fmat frames;
    uvec labels;
    cout<<"Reading data . . "<<endl;
    frames.load("../kws_sources/train_frames.bin");
    labels.load("../kws_sources/train_labels.bin");
    
    //  2.  Segment into phoneme classes keeps just the indices I think
    
    cout<<"Segmenting into phoneme classes"<<endl;
    
    uword numFrames = labels.n_elem;
    
    vector<uword> segments[NUMPHONES];
    for (uword i =0; i<numFrames; ++i) {
        segments[labels.at(i)-1].push_back(i);
        
    }

    uword subsetSize=100;
    uword index = 0;
    vector<uword> filtered_train;
    vector<uword> filtered_labels;
    segments[0].clear();
    for (uword i=1;i<NUMPHONES;i++){
        uvec dummyIter = shuffle(regspace<uvec> (0, 1, segments[i].size()-1));
        index=0;
        if (segments[i].size()!=0){
            while((index<subsetSize)&&(index<segments[i].size())){
                filtered_train.push_back(segments[i][dummyIter[index]]);
                filtered_labels.push_back(i);
                index++;
            }
            segments[i].clear();
        }
    }
    
    //  3.  Create the training set friends and foes
    

    cout<<"Creating the training set for Distance Learning"<<endl;
     

    vector<training> friendSet, foeSet;
     
    
    for (uword i=0; i<filtered_train.size()-1; i++) {
        for (uword j=i+1; j<filtered_train.size(); j++) {
     
            training dummyTrain;
            dummyTrain.x = i;
            dummyTrain.y = j;
     
            if (filtered_labels[i]==filtered_labels[j]) {
                dummyTrain.r=1;
                friendSet.push_back(dummyTrain);
            }
            else {
                dummyTrain.r=0;
                foeSet.push_back(dummyTrain);
            }
    
        }
    }
    
    //  3.1.    Train an autoencoder on the whole training set
    //  3.2.    Extend the friends set to match the size of foes set
    //  3.3.    Concatenate two sets into one DML training set
     
    

    //  4.  Initialize weights
    
    fmat W = eye<fmat>(HIDDEN,NUMPHONES) ;
    fmat del_W (size(W));
    fmat randMat = randu<fmat>(size(W));
    randMat = 0.2*(randMat-0.5);
    W += randMat;
    float b = 0.5;
    fvec x (NUMPHONES);
    fvec y (NUMPHONES);
    fvec h (HIDDEN); //Wx
    fvec g (HIDDEN); //Wy

    uword r;
    float d = 0.0; //h.g
    float z = 0.0; //sigmoid(d+b)
    
    //  5.  Learn distance
    
    cout<<"Learning distance . . ."<<endl;
    fvec J ;
    float orig_friend, orig_foe, new_friend, new_foe;
    orig_foe = 0.0;
    orig_friend = 0.0;
    uvec foe_iterator = shuffle(regspace <uvec> (0,1,foeSet.size()-1));
    uvec friend_iterator = shuffle(regspace <uvec> (0,1,friendSet.size()-1));
    float del_J = 0.0;
    bool flag1 = true;
    bool flag2 = true;
    string green="\033[1;32m";
    string colorend="\033[0m";
    string red="\033[4;31m";
    for (int i = 0; i<30; i++) {
        new_friend = 0.0;
        new_foe = 0.0;
        for (int j =0; j<friend_iterator.size(); j++) {
            x = frames.col(filtered_train[friendSet[friend_iterator[j]].x]);
            y = frames.col(filtered_train[friendSet[friend_iterator[j]].y]);
            r=1;
            h = W*x;
            g = W*y;
            d = dot(h,g) + b;
            z = sigmoid(d);
            new_friend += (1-z);
            del_J = (r-z);
            W += del_J*mu*(h*y.t()+g*x.t());
            b += mu2*del_J;
            //    J.push_back(-r*log(z)-(1-r)*log(1-z));
            if (flag1) {
                orig_friend+=(1-sigmoid(dot(x,y)+0.5));
            }
            
            
            x = frames.col(filtered_train[foeSet[foe_iterator[j]].x]);
            y = frames.col(filtered_train[foeSet[foe_iterator[j]].y]);
            r=0;
            h = W*x;
            g = W*y;
            d = dot(h,g) + b;
            z = sigmoid(d);
            new_foe += (1-z);
            del_J = (r-z);
            W += del_J*mu*(h*y.t()+g*x.t());
            b += mu2*del_J;
            //    J.push_back(-r*log(z)-(1-r)*log(1-z));
            if (flag1) {
                orig_foe+=(1-sigmoid(dot(x,y)+0.5));
            }
        }
            flag1 = false;
            flag2 = false;
            cout<<green<<"mean of friends from "<<orig_friend/friend_iterator.size()<<" to ->"<<new_friend/friend_iterator.size()<<colorend<<endl;
            cout<<red<<"mean of foes from    "<<orig_foe/friend_iterator.size()<<" to ->"<<new_foe/friend_iterator.size()<<colorend<<endl;
        
    }
  
    

    
    return 0;
    
}



float sigmoid (float x)
{
    
    return 1/(1+exp(-x));
    
}

void shuffleVec (int vectorsize, vector<int> &v)
{
    for (int i =0; i<vectorsize; i++) {
        v.push_back(i);
    }
    
    std::random_shuffle (v.begin(),v.end());
    
}