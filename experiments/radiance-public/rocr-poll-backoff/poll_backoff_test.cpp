#include "core/util/poll_backoff.h"
#include <climits>
#include <cstdlib>
#include <iostream>

using namespace rocr::core;

static void require(bool value) {
  if (!value) std::abort();
}

int main() {
  require(kPollNapFloorUs == 20);
  require(kPollNapCeilingMixedUs == 200);
  require(kPollNapCeilingUs == 2000);
  for (int ceiling : {kPollNapCeilingMixedUs, kPollNapCeilingUs, INT_MAX}) {
    int nap = kPollNapFloorUs;
    for (int i = 0; i < 64; ++i) {
      int next = NextPollNapUs(nap, ceiling);
      require(next >= nap && next <= ceiling);
      require(next == (static_cast<long long>(nap) * 2 > ceiling
                          ? ceiling : nap * 2));
      nap = next;
    }
    require(nap == ceiling);
    require(NextPollNapUs(nap, ceiling) == ceiling);
  }
  // Check every admitted duration in both production ceilings, including the
  // boundary at which doubling saturates. Assertions remain enabled in Release.
  for (int ceiling : {kPollNapCeilingMixedUs, kPollNapCeilingUs}) {
    for (int nap = kPollNapFloorUs; nap <= ceiling; ++nap) {
      require(NextPollNapUs(nap, ceiling) == std::min(nap * 2, ceiling));
    }
  }
  std::cout << "bounded backoff checks passed\n";
}
