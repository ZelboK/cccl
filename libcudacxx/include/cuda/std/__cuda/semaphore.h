// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_SEMAPHORE_H
#define _LIBCUDACXX___CUDA_SEMAPHORE_H

#include <cuda/std/detail/__config>
#include <cuda/atomic>
#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <thread_scope _Sco, ptrdiff_t __least_max_value = INT_MAX> 
class fair_semaphore {
  cuda::atomic<unsigned int, _Sco> tickets{0};
  cuda::atomic<unsigned int, _Sco> current{0};

  _CCCL_HOST_DEVICE void lock() {
    auto t = tickets.fetch_add(1, cuda::memory_order_relaxed);
    int wait_iterations = 1;
    const int max_iterations = 1024; 
    while (t != current.load(cuda::memory_order_acquire)) {
      for (int i = 0; i < wait_iterations; ++i) { // no-op 
      // will compiler optimize this out? 
      }
      
      wait_iterations = (wait_iterations * 2 >= max_iterations) ? max_iterations : wait_iterations * 2;
    }
  }
  _CCCL_HOST_DEVICE void unlock() { current.fetch_add(1, cuda::memory_order_release); }
};


template <thread_scope _Sco>
using binary_semaphore = fair_semaphore<_Sco, 1>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_SEMAPHORE_H
