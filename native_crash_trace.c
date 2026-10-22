/* Diagnostic preload only: record the native stack before stack-protector abort. */
#define _GNU_SOURCE
#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((noreturn)) void __stack_chk_fail(void)
{
    static const char message[] = "\nDIAGNOSTIC: __stack_chk_fail native backtrace\n";
    void *frames[64];
    write(STDERR_FILENO, message, sizeof(message) - 1);
    int count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    abort();
}
