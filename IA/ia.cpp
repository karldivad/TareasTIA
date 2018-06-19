#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <string.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <utility>

using namespace boost::numeric::ublas;

typedef int num;
typedef double cell;
typedef double number;
typedef matrix<cell> matriz;

void FillMe(matriz &empty){
  srand(time(NULL));
  for(auto i=empty.begin1(); i<empty.end1(); i++) for(auto j=i.begin(); j<i.end(); j++) *j=((double)rand()/(RAND_MAX));
}

matriz Create_Layer(num nNeurons, num nInputs){
  matriz tmp(nNeurons, nInputs);
  FillMe(tmp);
  return tmp;
}

matriz Net(matriz A, matriz B){
  return prod(trans(A),B);
}

void print(matriz A){
  std::cout << A << '\n';
}

std::vector<matriz> Create_Network(std::vector<int> nAll){
  std::vector<matriz> tmp;
  tmp.push_back(Create_Layer(nAll[0],1));
  for(int i=1; i<nAll.size(); i++) tmp.push_back(Create_Layer(nAll[i-1]+1,nAll[i]));
  return tmp;
}

void Sig(number &A){
  A = 1.0/(1.0+std::exp(-A));
}

void S(matriz &A){
  for(auto i=A.begin1(); i<A.end1(); i++) for(auto j=i.begin(); j<i.end(); j++) Sig(*j);
}

void printV(std::vector<matriz> NN){
  for(int i=0; i<NN.size(); i++) std::cout<< NN[i] <<'\n';
}

matriz AddRow1(matriz m){
  m.resize(m.size1()+1, m.size2(), true);
  for(int i=m.size1()-1; i>0; i--) memcpy(&m(i,0),&m(i-1,0),m.size2()*sizeof(number));
  m(0,0)=1.0;
  return m;
}

number Error(matriz results, number original){
  double sum = 0;
  for(int i=0; i<results.size2(); i++) if((int)original!=i) sum+=results(0,i)*results(0,i)/2; else sum+=(results(0,i)-1)*(results(0,i)-1)/2;
  return sum/results.size2();
}

std::pair<matriz,number> getResults(std::vector<matriz> &NN,std::vector<number> inputs, std::vector<matriz> &nSalidas){
  for(int i=0; i<NN[0].size1();i++) NN[0](i,0)=inputs[i];
  matriz tmp = Net(AddRow1(NN[0]),NN[1]);
  S(tmp);
  nSalidas.clear();
  nSalidas.push_back(trans(NN[0]));
  nSalidas.push_back(tmp);
  for(int i=2; i<NN.size(); i++) {tmp = Net(AddRow1(trans(tmp)),NN[i]); S(tmp); nSalidas.push_back(tmp);}
  return std::make_pair(tmp,Error(tmp,inputs.back()));
}

  std::vector<std::vector<number>> GetInputs(std::string archivinho){
  std::vector<std::vector<number>> inputs;

  std::ifstream file(archivinho);
  std::string line="";
  std::istringstream ss;
  std::string s="";

  while(getline(file,line)){
    ss.str(std::string());
    ss.clear();
    ss.str(line);
    inputs.push_back(std::vector<number>());
    while(getline(ss,s,',')) inputs.back().push_back(std::stod(s));
  }
  return inputs;
}

void ModifyWeights(std::vector<matriz> &NN, matriz outs, number target,std::vector<matriz> nSalidas, std::vector<std::vector<number>> &nDeltas){
  number alpha = 0.1;
  number t = 0;
  number sum = 0;
  number temp = 0;
  number delta = 0;
  int k=0;
  for(int mat=NN.size()-1; mat>0; mat--){
    for(int fila=0; fila<NN[mat].size1(); fila++){
      for(int col=0; col<NN[mat].size2(); col++){
        if(mat == NN.size()-1){
          if(col==target) t=1; else t=0;
          k++;
          //std::cout<<'a'<<'\n';
          delta = (t-outs(0,col)) * (outs(0,col))*(1-outs(0,col));
          //std::cout<<'b'<<'\n';
          NN[mat](fila,col) -= alpha*-delta*nSalidas[nSalidas.size()-2](0,col);
          //std::cout<<'c'<<'\n';
          nDeltas.back()[col] = delta;
          //std::cout<<'d'<<'\n';
          //std::cout<<NN[mat]<<'\n';
          //std::cout << k << ' ' << fila << ' ' << col << ' ' << t <<"   " << temp << "  " << col <<'\n';
          //std::cout << outs << '\n';
        }else{
          delta = 0;
          sum = 0;
          //std::cout<<NN[mat]<<'\n';
          //std::cout<< fila << ' ' << col << '\n';
          //std::cout<<'e'<<'\n';
          for(int neu=0; neu<NN[mat].size2(); neu++) sum+= nDeltas[mat-1][col]*NN[mat](fila,col);
          //std::cout<<'f'<<'\n';
          delta = sum * nSalidas[mat](0,col)*(1-nSalidas[mat](0,col));
          //std::cout<<'g'<<'\n';
          if(mat != 1) nDeltas[mat-1][col] = delta;
          //if(mat == 1) NN[mat](fila,col) -=alpha*sum*(1-nSalidas[mat](0,col))*nSalidas[mat](0,col)*trans(nSalidas[mat])(0,col);
          //else
          //std::cout<<'h'<<'\n';
          NN[mat](fila,col) -=alpha*-delta*nSalidas[mat](0,col);
        }
      }
    }
  }
}

bool compare(matriz a, matriz b){
  for(int i=0; i<a.size1(); i++){
    for(int j=0; j<a.size2(); j++){
      if(a(i,j) != b(i,j)) return false;
    }
  }
  return true;
}

int main (){
  std::vector<int> nAll = {4,8,3};
  auto inputs = GetInputs("iris.dat");
  auto outputs = GetInputs("test.dat");

  auto NN = Create_Network(nAll);

  std::vector<matriz> nSalidas(nAll.size()-1);
  std::vector<std::vector<number>> nDeltas(nAll.size()-2);
  for(int i=1; i<nAll.size()-1; i++) nDeltas[i-1].resize(nAll[i]);
  std::pair<matriz,number> results = getResults(NN,inputs[0],nSalidas);
  
  
  //printV(nSalidas);
  //std::cout<<"Red nn"<<'\n';

  //printV(NN);
  int iter=0;
  while(results.second>0.001 and iter<1000)
  {
    //std::cout << iter << ' ' << results.second << '\n';
    iter++;
    //std::cout<<"------------------"<<'\n';
    for(int i=0; i<inputs.size(); i++){
      results = getResults(NN,inputs[i],nSalidas);
      ModifyWeights(NN,results.first,inputs[i].back(),nSalidas,nDeltas);
      //if(i==0) std::cout << results.first << "   " << results.second << '\n';
      //else if(i==35) std::cout << results.first << "   " << results.second << '\n';
   }
    //std::cout << results.first << "   " << results.second << '\n';

}

  number confusion[3][3];
  for(int i=0; i<3; i++) for(int j=0; j<3; j++) confusion[i][j] = 0;
  matriz test0(1,3);
  test0(0,0)=1;
  test0(0,1)=0;
  test0(0,2)=0;

  std::cout<< test0 <<'\n';
  matriz test1(1,3);
  test1(0,1)=1;
  test1(0,0)=0;
  test1(0,2)=0;

  matriz test2(1,3);
  test2(0,2)=1;
  test0(0,1)=0;
  test0(0,0)=0;

  int rezagados = 0;
  int contador = 0;
  int totaln = 20;
  for(int i=0; i<outputs.size(); i++){
    results = getResults(NN,outputs[i],nSalidas);
 
    for(int k=0; k<results.first.size2(); k++) {if(results.first(0,k)>0.5) results.first(0,k) = 1; else results.first(0,k) = 0;}

    if(compare(results.first,test0) and contador<20) confusion[0][0]++;
    else if(compare(results.first,test1) and contador<20) confusion[0][1]++;
    else if(compare(results.first,test2) and contador<20) confusion[0][2]++;

    else if(compare(results.first,test0) and contador>20 and contador<40) confusion[1][0]++;
    else if(compare(results.first,test1) and contador>20 and contador<40) confusion[1][1]++;
    else if(compare(results.first,test2) and contador>20 and contador<40) confusion[1][2]++;

    else if(compare(results.first,test0) and contador>40 and contador<60) confusion[2][0]++;
    else if(compare(results.first,test1) and contador>40 and contador<60) confusion[2][1]++;
    else if(compare(results.first,test2) and contador>40 and contador<60) confusion[2][2]++;
    else rezagados ++;
    contador++;
  }

  std::cout<< '\t' << "  SETOSA  " << "  VERSICOLOR  " << "  VIRGINICA  " <<'\n';
  std::cout<< "SETOSA\t" << confusion[0][0]*100/totaln << '\t' << confusion[0][1]*100/totaln << '\t' << confusion[0][2]*100/totaln << '\t' << '\n';
  std::cout<< "VERSICOLOR\t" <<confusion[1][0]*100/totaln << '\t' << confusion[1][1]*100/totaln << '\t' << confusion[1][2]*100/totaln << '\t' << '\n';
  std::cout<< "VIRGINICA\t" <<confusion[2][0]*100/totaln << '\t' << confusion[2][1]*100/totaln << '\t' << confusion[2][2]*100/totaln << '\t' << '\n'; 
  std::cout<< "REZAGADOS: " << rezagados << '\n';
 return 0;
}



