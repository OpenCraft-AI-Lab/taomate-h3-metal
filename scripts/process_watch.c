#include <errno.h>
#include <libproc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

/* Read-only one-second samples; kernel lifetime peak is not a sampled peak. */
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char *end = NULL;
    long parsed = strtol(argv[1], &end, 10);
    if (!end || *end || parsed < 1 || parsed > INT32_MAX) return 2;
    int pid = (int)parsed;
    setvbuf(stdout, NULL, _IOLBF, 0);
    for (;;) {
        struct rusage_info_v4 process = {0};
        if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&process)) {
            fprintf(stderr, "sampling ended for pid %d (errno %d)\n", pid, errno);
            break;
        }
        struct xsw_usage swap = {0};
        size_t size = sizeof(swap);
        int swap_ok = !sysctlbyname("vm.swapusage", &swap, &size, NULL, 0);
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        printf("{\"unix_seconds\":%.6f,\"pid\":%d,\"rss_bytes\":%llu,"
               "\"footprint_bytes\":%llu,\"kernel_peak_footprint_bytes\":%llu,"
               "\"user_time_ns\":%llu,\"system_time_ns\":%llu,"
               "\"system_swap_bytes\":%llu,\"swap_read_ok\":%s}\n",
               (double)now.tv_sec + (double)now.tv_nsec / 1e9, pid,
               (unsigned long long)process.ri_resident_size,
               (unsigned long long)process.ri_phys_footprint,
               (unsigned long long)process.ri_lifetime_max_phys_footprint,
               (unsigned long long)process.ri_user_time,
               (unsigned long long)process.ri_system_time,
               (unsigned long long)swap.xsu_used, swap_ok ? "true" : "false");
        sleep(1);
    }
    return 0;
}
