#include <stdio.h>

int main(void) {
  int a = 246727561;
  float b = (float)a;
  printf("%f\n", b);
  float diff = 246727561.0 - b;
  printf("%f\n", diff);
}