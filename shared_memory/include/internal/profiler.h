#ifndef PROFILER_H
#define PROFILER_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef TRACY_ENABLE
#include "tracy/TracyC.h"

#define TRACY_ZONE_START TracyCZone(__tracy_ctx, 1)
#define TRACY_ZONE_END TracyCZoneEnd(__tracy_ctx)

#define TRACY_ALLOC(p,s) TracyCAlloc(p, s)
#define TRACY_FREE(p) TracyCFree(p)

#else

#define TRACY_ZONE_START
#define TRACY_ZONE_END

#define TRACY_ALLOC(p,s)
#define TRACY_FREE(p)

#endif

#ifdef __cplusplus
}
#endif

#endif  // PROFILER_H