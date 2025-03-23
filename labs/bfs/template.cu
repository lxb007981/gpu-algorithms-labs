#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 4096

// Number of warp queues per block
#define NUM_WARP_QUEUES 8
// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY (BQ_CAPACITY / NUM_WARP_QUEUES)

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queueing_kernel(unsigned int *nodePtrs,
                                           unsigned int *nodeNeighbors,
                                           unsigned int *nodeVisited,
                                           unsigned int *currLevelNodes,
                                           unsigned int *nextLevelNodes,
                                           unsigned int *numCurrLevelNodes,
                                           unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to global queue
  unsigned int tid      = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride   = blockDim.x * gridDim.x;
  unsigned int numNodes = *numCurrLevelNodes;
  for (unsigned int i = tid; i < numNodes; i += stride) {
    unsigned int node      = currLevelNodes[i];
    unsigned int nbrIdxEnd = nodePtrs[node + 1];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nbrIdxEnd; nbrIdx++) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        unsigned int oldNumNextLevelNodes = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[oldNumNextLevelNodes] = neighbor;
      }      
    }
  }
}

__global__ void gpu_block_queueing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE

  // Initialize shared memory queue (size should be BQ_CAPACITY)
  __shared__ unsigned int blockQueue[BQ_CAPACITY];
  __shared__ unsigned int blockQueueCounter;
  if (threadIdx.x == 0) {
    blockQueueCounter = 0;
  }
  __syncthreads();
  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to block queue
  // If full, add neighbor to global queue
  unsigned int tid      = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride   = blockDim.x * gridDim.x;
  unsigned int numNodes = *numCurrLevelNodes;
  for (unsigned int i = tid; i < numNodes; i += stride) {
    unsigned int node      = currLevelNodes[i];
    unsigned int nbrIdxEnd = nodePtrs[node + 1];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nbrIdxEnd; nbrIdx++) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        unsigned int oldBlockQueueCounter = atomicAdd(&blockQueueCounter, 1);
        if (oldBlockQueueCounter < BQ_CAPACITY) {
          blockQueue[oldBlockQueueCounter] = neighbor;
        } else {
          blockQueueCounter = BQ_CAPACITY;
          unsigned int oldNumNextLevelNodes = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[oldNumNextLevelNodes] = neighbor;
        }
      }      
    }
  }
  // Allocate space for block queue to go into global queue
  __syncthreads();
  __shared__ unsigned int globalQueueCounter;
  if (threadIdx.x == 0) {
    globalQueueCounter = atomicAdd(numNextLevelNodes, blockQueueCounter);
  }
  __syncthreads();
  // Store block queue in global queue
  for (unsigned int i = threadIdx.x; i < blockQueueCounter; i += blockDim.x) {
    nextLevelNodes[globalQueueCounter + i] = blockQueue[i];
  }
}

__global__ void gpu_warp_queueing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  // This version uses NUM_WARP_QUEUES warp queues of capacity
  // WQ_CAPACITY.  Be sure to interleave them as discussed in lecture.

  // Don't forget that you also need a block queue of capacity BQ_CAPACITY.

  // Initialize shared memory queues (warp and block)
  __shared__ unsigned int warpQueues[WQ_CAPACITY][NUM_WARP_QUEUES];
  __shared__ unsigned int warpQueueCounters[NUM_WARP_QUEUES];
  __shared__ unsigned int blockQueue[BQ_CAPACITY];
  __shared__ unsigned int blockQueueCounter;
  if (threadIdx.x == 0) {
    blockQueueCounter = 0;
    for (unsigned int i = 0; i < NUM_WARP_QUEUES; i++) {
      warpQueueCounters[i] = 0;
    }
  }
  __syncthreads();
  // Loop over all nodes in the current level
  // Loop over all neighbors of the node
  // If neighbor hasn't been visited yet
  // Add neighbor to the queue
  // If full, add neighbor to block queue
  // If full, add neighbor to global queue
  unsigned int tid      = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride   = blockDim.x * gridDim.x;
  unsigned int numNodes = *numCurrLevelNodes;
  for (unsigned int i = tid; i < numNodes; i += stride) {
    unsigned int node      = currLevelNodes[i];
    unsigned int nbrIdxEnd = nodePtrs[node + 1];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nbrIdxEnd; nbrIdx++) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      if (atomicCAS(&nodeVisited[neighbor], 0, 1) == 0) {
        unsigned int warpQueueIdx = neighbor % NUM_WARP_QUEUES;
        unsigned int oldWarpQueueCounter = atomicAdd(&warpQueueCounters[warpQueueIdx], 1);
        if (oldWarpQueueCounter < WQ_CAPACITY) {
          warpQueues[oldWarpQueueCounter][warpQueueIdx] = neighbor;
        } else {
          warpQueueCounters[warpQueueIdx] = WQ_CAPACITY;
          unsigned int oldBlockQueueCounter = atomicAdd(&blockQueueCounter, 1);
          if (oldBlockQueueCounter < BQ_CAPACITY) {
            blockQueue[oldBlockQueueCounter] = neighbor;
          } else {
            blockQueueCounter = BQ_CAPACITY;
            unsigned int oldNumNextLevelNodes = atomicAdd(numNextLevelNodes, 1);
            nextLevelNodes[oldNumNextLevelNodes] = neighbor;
          }
        }
      }      
    }
  }
  // Allocate space for warp queue to go into block queue
  __syncthreads();
  __shared__ unsigned int blockQueueCounter2;
  if (threadIdx.x == 0) {
    blockQueueCounter2 = atomicAdd(&blockQueueCounter, NUM_WARP_QUEUES * WQ_CAPACITY);
  }
  __syncthreads();
  // Store warp queues in block queue (use one warp or one thread per queue)
  // Add any nodes that don't fit (remember, space was allocated above)
  //    to the global queue
  if (threadIdx.x < NUM_WARP_QUEUES) {
    unsigned int warpQueueCounter = warpQueueCounters[threadIdx.x];
    for (unsigned int i = 0; i < warpQueueCounter; i++) {
      unsigned int oldBlockQueueCounter = atomicAdd(&blockQueueCounter2, 1);
      if (oldBlockQueueCounter < BQ_CAPACITY) {
        blockQueue[oldBlockQueueCounter] = warpQueues[i][threadIdx.x];
      } else {
        blockQueueCounter2 = BQ_CAPACITY;
        unsigned int oldNumNextLevelNodes = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[oldNumNextLevelNodes] = warpQueues[i][threadIdx.x];
      }
    }
  }

  // Saturate block queue counter (too large if warp queues overflowed)
  // Allocate space for block queue to go into global queue
  __syncthreads();
  __shared__ unsigned int globalQueueCounter;
  if (threadIdx.x == 0) {
    //blockQueueCounter2 = min(blockQueueCounter2, BQ_CAPACITY);
    globalQueueCounter = atomicAdd(numNextLevelNodes, blockQueueCounter2);
  }
  __syncthreads();
  // Store block queue in global queue
  for (unsigned int i = threadIdx.x; i < blockQueueCounter2; i += blockDim.x) {
    nextLevelNodes[globalQueueCounter + i] = blockQueue[i];
  }
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors, unsigned int *nodeVisited, unsigned int *currLevelNodes,
                         unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queueing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
                                                        numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors, unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queueing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
                                                       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors, unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queueing_kernel<<<numBlocks, BLOCK_SIZE>>>(nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
                                                      numCurrLevelNodes, numNextLevelNodes);
}
