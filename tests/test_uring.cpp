#include <cerrno>
#include <cstring>
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(SYS_io_uring_setup)
#define PIPEANN_IO_URING_SETUP SYS_io_uring_setup
#elif defined(__NR_io_uring_setup)
#define PIPEANN_IO_URING_SETUP __NR_io_uring_setup
#endif

static int setup_ring(unsigned flags) {
#ifndef PIPEANN_IO_URING_SETUP
  return ENOSYS;
#else
  io_uring_params params;
  std::memset(&params, 0, sizeof(params));
  params.flags = flags;
  int fd = static_cast<int>(syscall(PIPEANN_IO_URING_SETUP, 2, &params));
  if (fd < 0) {
    return errno;
  }
  close(fd);
  return 0;
#endif
}

int main() {
  return setup_ring(0);
}
