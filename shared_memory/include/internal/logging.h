// A small logging header

#include <string.h>

#ifdef ENABLE_LOGS

#define LIB_PRINT_PREFIX "[SHM] "

// Source -
// https://stackoverflow.com/questions/8487986/file-macro-shows-full-path Posted
// by red1ynx Retrieved 11/5/2025, License - CC BY-SA 3.0
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define LOG_LINE(fmt, args...) \
  LIB_PRINT_PREFIX "   %s:%i " fmt, __FILENAME__, __LINE__, args
#define LOG_INF(fmt, args...) fprintf(stdout, LOG_LINE(fmt, args))
#define LOG_ERR(fmt, args...) fprintf(stderr, LOG_LINE(fmt, args))

#else

#define LOG_INF(args...)
#define LOG_ERR(args...)

#endif

// Source - https://stackoverflow.com/a
// Posted by ideasman42, modified by community. See post 'Timeline' for change
// history Retrieved 2025-11-21, License - CC BY-SA 3.0

/* --- PRINTF_BYTE_TO_BINARY macro's --- */
#define PRINTF_BINARY_PATTERN_INT8 "%c%c%c%c%c%c%c%c "
#define PRINTF_BYTE_TO_BINARY_INT8(i)                             \
  (((i) & 0x80ll) ? '1' : '0'), (((i) & 0x40ll) ? '1' : '0'),     \
      (((i) & 0x20ll) ? '1' : '0'), (((i) & 0x10ll) ? '1' : '0'), \
      (((i) & 0x08ll) ? '1' : '0'), (((i) & 0x04ll) ? '1' : '0'), \
      (((i) & 0x02ll) ? '1' : '0'), (((i) & 0x01ll) ? '1' : '0')

#define PRINTF_BINARY_PATTERN_INT16 \
  PRINTF_BINARY_PATTERN_INT8 PRINTF_BINARY_PATTERN_INT8
#define PRINTF_BYTE_TO_BINARY_INT16(i) \
  PRINTF_BYTE_TO_BINARY_INT8((i) >> 8), PRINTF_BYTE_TO_BINARY_INT8(i)
#define PRINTF_BINARY_PATTERN_INT32 \
  PRINTF_BINARY_PATTERN_INT16 PRINTF_BINARY_PATTERN_INT16
#define PRINTF_BYTE_TO_BINARY_INT32(i) \
  PRINTF_BYTE_TO_BINARY_INT16((i) >> 16), PRINTF_BYTE_TO_BINARY_INT16(i)
#define PRINTF_BINARY_PATTERN_INT64 \
  PRINTF_BINARY_PATTERN_INT32 PRINTF_BINARY_PATTERN_INT32
#define PRINTF_BYTE_TO_BINARY_INT64(i) \
  PRINTF_BYTE_TO_BINARY_INT32((i) >> 32), PRINTF_BYTE_TO_BINARY_INT32(i)
/* --- end macros --- */
