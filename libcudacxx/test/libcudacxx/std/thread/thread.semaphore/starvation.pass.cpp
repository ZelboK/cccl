#include <cuda.h>
#include <cstdio>
#include <cuda/semaphore>

struct semaphore_lock {
    cuda::binary_semaphore<cuda::thread_scope_block> s{1};
    __device__ void lock() {
        s.acquire();
    }
    __device__ void unlock() {
       s.release();
    }
};

__device__ semaphore_lock l{};
__device__ int mask = 0;

__global__ void reproducer() {
  l.lock();
  bool cont = false;
  do {
    l.unlock();
    cont = atomicAdd_block(&mask, threadIdx.x) == 0;
    l.lock();
  } while (cont);
  l.unlock();
}