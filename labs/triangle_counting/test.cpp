#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <vector>

bool binary_search(const uint32_t *const edgeDst, uint32_t start, uint32_t end, uint32_t target) {
  while (start <= end) {
    uint32_t mid = start + (end - start) / 2;
    if (edgeDst[mid] == target) {
      return true;
    } else if (edgeDst[mid] < target) {
      start = mid + 1;
    } else {
      end = mid - 1;
    }
  }
  return false;
}

int main() {
  // test binary search
  uint32_t edgeDst[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  uint32_t start = 0;
  uint32_t end = 10;
  uint32_t target = 23;
  bool result = binary_search(edgeDst, start, end, target);
  printf("result: %d\n", result);
}