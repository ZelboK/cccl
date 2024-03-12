#ifndef _LIBCUDACXX___FAIR_SEMAPHORE_H
#define _LIBCUDACXX___FAIR_SEMAPHORE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__
#include <cuda/atomic>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <thread_scope _Sco, ptrdiff_t __least_max_value = INT_MAX>
struct __fair_semaphore {
  cuda::atomic<unsigned int, _Sco> tickets{0};
  cuda::atomic<unsigned int, _Sco> current{0};

  _LIBCUDACXX_INLINE_VISIBILITY static constexpr ptrdiff_t max() noexcept {
    return __least_max_value;  
  }

  void lock() {
    auto t = tickets.fetch_add(1, cuda::memory_order_relaxed);
    int wait_iterations = 1;
    const int max_iterations = 1024;
    while (t != current.load(cuda::memory_order_acquire)) {
      for (int i = 0; i < wait_iterations; i++) { // no-op
      }
      wait_iterations = min(wait_iterations * 2, max_iterations);
    }
  }
  void unlock() { current.fetch_add(1, cuda::memory_order_release); }
};


_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FAIR__SEMAPHORE_H
