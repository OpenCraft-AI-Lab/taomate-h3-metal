#include <errno.h>
#include <libproc.h>
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

/* Samples only the requested PID. The guarded Python launcher owns termination. */
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char *end = NULL;
    long parsed = strtol(argv[1], &end, 10);
    if (!end || *end || parsed < 1 || parsed > INT32_MAX) return 2;
    int pid = (int)parsed;
    mach_port_t host = mach_host_self();
    vm_size_t page = 0;
    if (host_page_size(host, &page) != KERN_SUCCESS) return 3;
    setvbuf(stdout, NULL, _IOLBF, 0);
    for (;;) {
        struct rusage_info_v4 process = {0};
        if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&process)) return 0;
        struct xsw_usage swap = {0};
        size_t size = sizeof(swap);
        int swap_ok = !sysctlbyname("vm.swapusage", &swap, &size, NULL, 0);
        vm_statistics64_data_t vm = {0};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        int vm_ok = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm,
                                      &count) == KERN_SUCCESS;
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        /* free+inactive is a conservative reclaimable estimate, not RAM-free. */
        printf("{\"unix_seconds\":%.6f,\"pid\":%d,\"rss_bytes\":%llu,"
               "\"footprint_bytes\":%llu,\"kernel_peak_footprint_bytes\":%llu,"
               "\"available_estimate_bytes\":%llu,\"system_swap_bytes\":%llu,"
               "\"sample_ok\":%s}\n",
               (double)now.tv_sec + (double)now.tv_nsec / 1e9, pid,
               (unsigned long long)process.ri_resident_size,
               (unsigned long long)process.ri_phys_footprint,
               (unsigned long long)process.ri_lifetime_max_phys_footprint,
               (unsigned long long)(vm.free_count + vm.inactive_count) * page,
               (unsigned long long)swap.xsu_used, vm_ok && swap_ok ? "true" : "false");
        struct timespec delay = {.tv_sec = 0, .tv_nsec = 250000000};
        nanosleep(&delay, NULL);
    }
}
