// ml_mesh patch: serial stand-in for pss (needs TBB tasks).
#pragma once
#include <algorithm>
namespace pss {
template <typename It, typename C> void parallel_stable_sort(It a, It b, C c) { std::stable_sort(a, b, c); }
template <typename It> void parallel_stable_sort(It a, It b) { std::stable_sort(a, b); }
}
