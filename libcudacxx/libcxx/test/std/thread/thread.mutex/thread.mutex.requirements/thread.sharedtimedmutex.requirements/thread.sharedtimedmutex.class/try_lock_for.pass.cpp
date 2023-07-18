//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03, c++11

// Threading tests are notoriously flaky in libcxx mode so disable them
// UNSUPPORTED: true

// <shared_mutex>

// class shared_timed_mutex;

// template <class Rep, class Period>
//     bool try_lock_for(const chrono::duration<Rep, Period>& rel_time);

#include <shared_mutex>
#include <thread>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

std::shared_timed_mutex m;

typedef std::chrono::steady_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;


ms WaitTime = ms(250);

// Thread sanitizer causes more overhead and will sometimes cause this test
// to fail. To prevent this we give Thread sanitizer more time to complete the
// test.
#if !defined(TEST_HAS_SANITIZERS)
ms Tolerance = ms(50);
#else
ms Tolerance = ms(50 * 5);
#endif

void f1()
{
    time_point t0 = Clock::now();
    assert(m.try_lock_for(WaitTime + Tolerance) == true);
    time_point t1 = Clock::now();
    m.unlock();
    ns d = t1 - t0 - WaitTime;
    assert(d < Tolerance);  // within tolerance
}

void f2()
{
    time_point t0 = Clock::now();
    assert(m.try_lock_for(WaitTime) == false);
    time_point t1 = Clock::now();
    ns d = t1 - t0 - WaitTime;
    assert(d < Tolerance);  // within tolerance
}

int main(int, char**)
{
    {
        m.lock();
        std::thread t(f1);
        std::this_thread::sleep_for(WaitTime);
        m.unlock();
        t.join();
    }
    {
        m.lock();
        std::thread t(f2);
        std::this_thread::sleep_for(WaitTime + Tolerance);
        m.unlock();
        t.join();
    }

  return 0;
}