#include <iomanip>
#include <iostream>
using namespace std;
int main(void) {
  int a   = 246727561;
  double b = static_cast<double>(a);
  cout << std::fixed << std::setprecision(10) << b << endl;
  double diff = 246727561.0 - b;
  cout << diff << endl;
}