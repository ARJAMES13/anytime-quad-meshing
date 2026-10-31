// ml_mesh patch: serial stand-in for TBB 4.x (old static TBB crashes under MinGW).
#pragma once
#include <algorithm>
#ifdef _WIN32
#include <intrin.h>
#endif
#include <cstddef>
#include <functional>
#include <new>
#include <queue>
#include <vector>
namespace tbb {
template <typename T> class blocked_range {
    T b_, e_; size_t g_;
public:
    typedef T const_iterator;
    blocked_range(T b, T e, size_t g = 1) : b_(b), e_(e), g_(g) {}
    T begin() const { return b_; }
    T end() const { return e_; }
    size_t size() const { return size_t(e_ - b_); }
    bool empty() const { return !(b_ < e_); }
    size_t grainsize() const { return g_; }
};
template <typename R, typename F> void parallel_for(const R &r, const F &f) { if (!r.empty()) f(r); }
template <typename I, typename F> void parallel_for(I a, I b, const F &f) { for (I i = a; i < b; ++i) f(i); }
template <typename R, typename V, typename F, typename Red>
V parallel_reduce(const R &r, const V &id, const F &f, const Red &) { return r.empty() ? id : f(r, id); }
template <typename R, typename V, typename F, typename Red>
V parallel_deterministic_reduce(const R &r, const V &id, const F &f, const Red &) { return r.empty() ? id : f(r, id); }
template <typename It> void parallel_sort(It a, It b) { std::sort(a, b); }
template <typename It, typename C> void parallel_sort(It a, It b, const C &c) { std::sort(a, b, c); }
class spin_mutex {
public:
    void lock() {}
    void unlock() {}
    bool try_lock() { return true; }
public:
    class scoped_lock {
    public:
        scoped_lock() {}
        explicit scoped_lock(spin_mutex &) {}
        void acquire(spin_mutex &) {}
        void release() {}
    };
};
class task_scheduler_init {
public:
    static const int automatic = -1;
    task_scheduler_init(int = automatic) {}
};
template <typename T, typename A = std::allocator<T>> using concurrent_vector = std::vector<T, A>;
template <typename T, typename C = std::less<T>> class concurrent_priority_queue {
    std::priority_queue<T, std::vector<T>, C> q_;
public:
    void push(const T &v) { q_.push(v); }
    bool try_pop(T &v) { if (q_.empty()) return false; v = q_.top(); q_.pop(); return true; }
    bool empty() const { return q_.empty(); }
    size_t size() const { return q_.size(); }
};
// Only to compile bvh.cpp's task class; the build path is patched to run serially.
class task {
public:
    virtual ~task() {}
    virtual task *execute() = 0;
    void *allocate_continuation() { return ::operator new(4096); }
    void *allocate_child() { return ::operator new(4096); }
    static void *allocate_root() { return ::operator new(4096); }
    void set_ref_count(int) {}
    void spawn(task &) {}
    void recycle_as_child_of(task &) {}
};
class empty_task : public task { public: task *execute() { return nullptr; } };
}  // namespace tbb
