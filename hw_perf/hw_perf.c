#include "hw_perf.h"

/// Get lightweight PET mode (not in kperf.framework).
static int kperf_lightweight_pet_get(u32 *enabled) {
  if (!enabled)
    return -1;
  usize size = 4;
  return sysctlbyname("kperf.lightweight_pet", enabled, &size, NULL, 0);
}

/// Set lightweight PET mode (not in kperf.framework).
static int kperf_lightweight_pet_set(u32 enabled) {
  return sysctlbyname("kperf.lightweight_pet", NULL, NULL, &enabled, 4);
}

/// Error description for kpep_config_error_code.
static const char *kpep_config_error_names[KPEP_CONFIG_ERROR_MAX] = {
    "none",
    "invalid argument",
    "out of memory",
    "I/O",
    "buffer too small",
    "current system unknown",
    "database path invalid",
    "database not found",
    "database architecture unsupported",
    "database version unsupported",
    "database corrupt",
    "event not found",
    "conflicting events",
    "all counters must be forced",
    "event unavailable",
    "check errno"};

/// Error description.
static const char *kpep_config_error_desc(int code) {
  if (0 <= code && code < KPEP_CONFIG_ERROR_MAX) {
    return kpep_config_error_names[code];
  }
  return "unknown error";
}

static bool lib_inited = false;
static bool lib_has_err = false;
static char lib_err_msg[256];

static void *lib_handle_kperf = NULL;
static void *lib_handle_kperfdata = NULL;

static void lib_deinit(void) {
  lib_inited = false;
  lib_has_err = false;
  if (lib_handle_kperf)
    dlclose(lib_handle_kperf);
  if (lib_handle_kperfdata)
    dlclose(lib_handle_kperfdata);
  lib_handle_kperf = NULL;
  lib_handle_kperfdata = NULL;
  for (usize i = 0; i < lib_nelems(lib_symbols_kperf); i++) {
    const lib_symbol *symbol = &lib_symbols_kperf[i];
    *symbol->impl = NULL;
  }
  for (usize i = 0; i < lib_nelems(lib_symbols_kperfdata); i++) {
    const lib_symbol *symbol = &lib_symbols_kperfdata[i];
    *symbol->impl = NULL;
  }
}

static bool lib_init(void) {
#define return_err()                                                           \
  do {                                                                         \
    lib_deinit();                                                              \
    lib_inited = true;                                                         \
    lib_has_err = true;                                                        \
    return false;                                                              \
  } while (false)

  if (lib_inited)
    return !lib_has_err;

  // load dynamic library
  lib_handle_kperf = dlopen(lib_path_kperf, RTLD_NOW);
  if (!lib_handle_kperf) {
    snprintf(lib_err_msg, sizeof(lib_err_msg),
             "Failed to load kperf.framework, message: %s.", dlerror());
    return_err();
  }
  lib_handle_kperfdata = dlopen(lib_path_kperfdata, RTLD_LAZY);
  if (!lib_handle_kperfdata) {
    snprintf(lib_err_msg, sizeof(lib_err_msg),
             "Failed to load kperfdata.framework, message: %s.", dlerror());
    return_err();
  }

  // load symbol address from dynamic library
  for (usize i = 0; i < lib_nelems(lib_symbols_kperf); i++) {
    const lib_symbol *symbol = &lib_symbols_kperf[i];
    *symbol->impl = dlsym(lib_handle_kperf, symbol->name);
    if (!*symbol->impl) {
      snprintf(lib_err_msg, sizeof(lib_err_msg),
               "Failed to load kperf function: %s.", symbol->name);
      return_err();
    }
  }
  for (usize i = 0; i < lib_nelems(lib_symbols_kperfdata); i++) {
    const lib_symbol *symbol = &lib_symbols_kperfdata[i];
    *symbol->impl = dlsym(lib_handle_kperfdata, symbol->name);
    if (!*symbol->impl) {
      snprintf(lib_err_msg, sizeof(lib_err_msg),
               "Failed to load kperfdata function: %s.", symbol->name);
      return_err();
    }
  }

  lib_inited = true;
  lib_has_err = false;
  return true;

#undef return_err
}

// -----------------------------------------------------------------------------
// kdebug utils
// -----------------------------------------------------------------------------

/// Clean up trace buffers and reset ktrace/kdebug/kperf.
/// @return 0 on success.
static int kdebug_reset(void) {
  int mib[3] = {CTL_KERN, KERN_KDEBUG, KERN_KDREMOVE};
  return sysctl(mib, 3, NULL, NULL, NULL, 0);
}

/// Disable and reinitialize the trace buffers.
/// @return 0 on success.
static int kdebug_reinit(void) {
  int mib[3] = {CTL_KERN, KERN_KDEBUG, KERN_KDSETUP};
  return sysctl(mib, 3, NULL, NULL, NULL, 0);
}

/// Set debug filter.
static int kdebug_setreg(kd_regtype *kdr) {
  int mib[3] = {CTL_KERN, KERN_KDEBUG, KERN_KDSETREG};
  usize size = sizeof(kd_regtype);
  return sysctl(mib, 3, kdr, &size, NULL, 0);
}

/// Set maximum number of trace entries (kd_buf).
/// Only allow allocation up to half the available memory (sane_size).
/// @return 0 on success.
static int kdebug_trace_setbuf(int nbufs) {
  int mib[4] = {CTL_KERN, KERN_KDEBUG, KERN_KDSETBUF, nbufs};
  return sysctl(mib, 4, NULL, NULL, NULL, 0);
}

/// Enable or disable kdebug trace.
/// Trace buffer must already be initialized.
/// @return 0 on success.
static int kdebug_trace_enable(bool enable) {
  int mib[4] = {CTL_KERN, KERN_KDEBUG, KERN_KDENABLE, enable};
  return sysctl(mib, 4, NULL, 0, NULL, 0);
}

/// Retrieve trace buffer information from kernel.
/// @return 0 on success.
static int kdebug_get_bufinfo(kbufinfo_t *info) {
  if (!info)
    return -1;
  int mib[3] = {CTL_KERN, KERN_KDEBUG, KERN_KDGETBUF};
  size_t needed = sizeof(kbufinfo_t);
  return sysctl(mib, 3, info, &needed, NULL, 0);
}

/// Retrieve trace buffers from kernel.
/// @param buf Memory to receive buffer data, array of `kd_buf`.
/// @param len Length of `buf` in bytes.
/// @param count Number of trace entries (kd_buf) obtained.
/// @return 0 on success.
static int kdebug_trace_read(void *buf, usize len, usize *count) {
  if (count)
    *count = 0;
  if (!buf || !len)
    return -1;

  // Note: the input and output units are not the same.
  // input: bytes
  // output: number of kd_buf
  int mib[3] = {CTL_KERN, KERN_KDEBUG, KERN_KDREADTR};
  int ret = sysctl(mib, 3, buf, &len, NULL, 0);
  if (ret != 0)
    return ret;
  *count = len;
  return 0;
}

/// Block until there are new buffers filled or `timeout_ms` have passed.
/// @param timeout_ms timeout milliseconds, 0 means wait forever.
/// @param suc set true if new buffers filled.
/// @return 0 on success.
static int kdebug_wait(usize timeout_ms, bool *suc) {
  if (timeout_ms == 0)
    return -1;
  int mib[3] = {CTL_KERN, KERN_KDEBUG, KERN_KDBUFWAIT};
  usize val = timeout_ms;
  int ret = sysctl(mib, 3, NULL, &val, NULL, 0);
  if (suc)
    *suc = !!val;
  return ret;
}

#define EVENT_NAME_MAX 8
typedef struct {
  const char *alias;                 /// name for print
  const char *names[EVENT_NAME_MAX]; /// name from pmc db
} event_alias;

/// Event names from /usr/share/kpep/<name>.plist
static const event_alias profile_events[] = {
    {"cycles",
     {
         "FIXED_CYCLES",            // Apple A7-A15
         "CPU_CLK_UNHALTED.THREAD", // Intel Core 1th-10th
         "CPU_CLK_UNHALTED.CORE",   // Intel Yonah, Merom
     }},
    {"instructions",
     {
         "FIXED_INSTRUCTIONS", // Apple A7-A15
         "INST_RETIRED.ANY"    // Intel Yonah, Merom, Core 1th-10th
     }},
    {"branches",
     {
         "INST_BRANCH",                  // Apple A7-A15
         "BR_INST_RETIRED.ALL_BRANCHES", // Intel Core 1th-10th
         "INST_RETIRED.ANY",             // Intel Yonah, Merom
     }},
    {"branch-misses",
     {
         "BRANCH_MISPRED_NONSPEC",       // Apple A7-A15, since iOS 15, macOS 12
         "BRANCH_MISPREDICT",            // Apple A7-A14
         "BR_MISP_RETIRED.ALL_BRANCHES", // Intel Core 2th-10th
         "BR_INST_RETIRED.MISPRED",      // Intel Yonah, Merom
     }},
    {"l1d-tlb-miss",
     {
         "L1D_TLB_MISS",
     }},
    {"l2-tlb-miss",
     {
        "L2_TLB_MISS_DATA",
     }},
    {"l1d-miss",
     {
        "L1D_CACHE_MISS_LD",
     }},
    {"mmu-table-walk",
     {
        "MMU_TABLE_WALK_DATA",
     }},
};

static kpep_event *get_event(kpep_db *db, const event_alias *alias) {
  for (usize j = 0; j < EVENT_NAME_MAX; j++) {
    const char *name = alias->names[j];
    if (!name)
      break;
    kpep_event *ev = NULL;
    if (kpep_db_event(db, name, &ev) == 0) {
      return ev;
    }
  }
  return NULL;
}

static double get_timestamp(void) {
  struct timeval now;
  gettimeofday(&now, NULL);
  return (double)now.tv_sec + (double)now.tv_usec / (1000.0 * 1000.0);
}

void usage(const char *exec) {
  printf("%s [-p target_pid] [-s sample_period] -- cmd ...\n", exec);
}

int main(int argc, char *const argv[]) {
  /// Target process pid, -1 for all thread.
  int target_pid = -1;

  /// pid of the process started using -- syntax
  int cmd_pid = -1;

  /// Profile sampler period in seconds (default 10ms).
  double sample_period = 0.001;

  char *const *exec_argv;
  int exec_argc;

  for (usize i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;

    } else if (strcmp(argv[i], "-p") == 0) {
      if (argc <= i + 1) {
        fprintf(stderr, "ERROR: -p must have value \n");
        usage(argv[0]);
        return 1;
      }
      int value = atoi(argv[i + 1]);
      if (value > 0) {
        target_pid = value;
        i++;
      }

    } else if (strcmp(argv[i], "-s") == 0) {
      if (argc <= i + 1) {
        fprintf(stderr, "ERROR: -s must have value \n");
        usage(argv[0]);
        return 1;
      }
      double value = atof(argv[i + 1]);
      if (value > 0.0) {
        sample_period = value;
        i++;
      }

    } else if (strcmp(argv[i], "--") == 0) {
      if (argc <= i + 1) {
        fprintf(stderr, "ERROR: -- must have at least one argument \n");
        usage(argv[0]);
        return 1;
      }
      exec_argc = argc - i - 1;
      exec_argv = &argv[i + 1];
      break;

    } else {
      fprintf(stderr, "ERROR: Unknown option '%s'\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  if (exec_argc <= 0) {
    fprintf(stderr,
            "ERROR: tracing process should be provided using [--] syntax\n");
    return 1;
  }

  if (exec_argc > 0) {
    int pid = fork();
    if (pid == -1) {
      fprintf(stderr, "ERROR: fork() failed\n");
      return 1;
    } else if (pid == 0) {
      // child
      if (execvp(exec_argv[0], exec_argv) == -1) {
        fprintf(stderr, "ERROR: unable to execute process\n");
        return 1;
      }
    } else {
      cmd_pid = pid;
      // if target_pid is provided by the use we'll use cmd process only
      // as a mean to wait for a given amount of time
      if (target_pid == -1) {
        target_pid = pid;
      }
    }
  }

  int ret = 0;

  // load dylib
  if (!lib_init()) {
    printf("Error: %s\n", lib_err_msg);
    return 1;
  }

  // check permission
  int force_ctrs = 0;
  if (kpc_force_all_ctrs_get(&force_ctrs)) {
    printf("Permission denied, xnu/kpc requires root privileges.\n");
    return 1;
  }

  // load pmc db
  kpep_db *db = NULL;
  if ((ret = kpep_db_create(NULL, &db))) {
    printf("Error: cannot load pmc database: %d.\n", ret);
    return 1;
  }
  printf("loaded db: %s (%s)\n", db->name, db->marketing_name);
  printf("number of fixed counters: %zu\n", db->fixed_counter_count);
  printf("number of configurable counters: %zu\n", db->config_counter_count);
  printf("CPU tick frequency: %llu\n",
         (unsigned long long)kperf_tick_frequency());

  // create a config
  kpep_config *cfg = NULL;
  if ((ret = kpep_config_create(db, &cfg))) {
    printf("Failed to create kpep config: %d (%s).\n", ret,
           kpep_config_error_desc(ret));
    return 1;
  }
  if ((ret = kpep_config_force_counters(cfg))) {
    printf("Failed to force counters: %d (%s).\n", ret,
           kpep_config_error_desc(ret));
    return 1;
  }

  // get events
  const usize ev_count = sizeof(profile_events) / sizeof(profile_events[0]);
  kpep_event *ev_arr[ev_count];
  for (usize i = 0; i < ev_count; i++) {
    const event_alias *alias = profile_events + i;
    ev_arr[i] = get_event(db, alias);
    if (!ev_arr[i]) {
      printf("Cannot find event: %s.\n", alias->alias);
      return 1;
    }
  }

  // add event to config
  for (usize i = 0; i < ev_count; i++) {
    kpep_event *ev = ev_arr[i];
    if ((ret = kpep_config_add_event(cfg, &ev, 0, NULL))) {
      printf("Failed to add event: %d (%s).\n", ret,
             kpep_config_error_desc(ret));
      return 1;
    }
  }

  // prepare buffer and config
  u32 classes = 0;
  usize reg_count = 0;
  kpc_config_t regs[KPC_MAX_COUNTERS] = {0};
  usize counter_map[KPC_MAX_COUNTERS] = {0};
  if ((ret = kpep_config_kpc_classes(cfg, &classes))) {
    printf("Failed get kpc classes: %d (%s).\n", ret,
           kpep_config_error_desc(ret));
    return 1;
  }
  if ((ret = kpep_config_kpc_count(cfg, &reg_count))) {
    printf("Failed get kpc count: %d (%s).\n", ret,
           kpep_config_error_desc(ret));
    return 1;
  }
  if ((ret = kpep_config_kpc_map(cfg, counter_map, sizeof(counter_map)))) {
    printf("Failed get kpc map: %d (%s).\n", ret, kpep_config_error_desc(ret));
    return 1;
  }
  if ((ret = kpep_config_kpc(cfg, regs, sizeof(regs)))) {
    printf("Failed get kpc registers: %d (%s).\n", ret,
           kpep_config_error_desc(ret));
    return 1;
  }

  // set config to kernel
  if ((ret = kpc_force_all_ctrs_set(1))) {
    printf("Failed force all ctrs: %d.\n", ret);
    return 1;
  }
  if ((classes & KPC_CLASS_CONFIGURABLE_MASK) && reg_count) {
    if ((ret = kpc_set_config(classes, regs))) {
      printf("Failed set kpc config: %d.\n", ret);
      return 1;
    }
  }

  u32 counter_count = kpc_get_counter_count(classes);
  if (counter_count == 0) {
    printf("Failed no counter\n");
    return 1;
  }

  // start counting
  if ((ret = kpc_set_counting(classes))) {
    printf("Failed set counting: %d.\n", ret);
    return 1;
  }
  if ((ret = kpc_set_thread_counting(classes))) {
    printf("Failed set thread counting: %d.\n", ret);
    return 1;
  }

  // action id and timer id
  u32 actionid = 1;
  u32 timerid = 1;

  // alloc action and timer ids
  if ((ret = kperf_action_count_set(KPERF_ACTION_MAX))) {
    printf("Failed set action count: %d.\n", ret);
  }
  if ((ret = kperf_timer_count_set(KPERF_TIMER_MAX))) {
    printf("Failed set timer count: %d.\n", ret);
  }

  // set what to sample: PMC per thread
  if ((ret = kperf_action_samplers_set(actionid, KPERF_SAMPLER_PMC_THREAD))) {
    printf("Failed set sampler type: %d.\n", ret);
  }
  // set filter process
  if ((ret = kperf_action_filter_set_by_pid(actionid, target_pid))) {
    printf("Failed set filter pid: %d.\n", ret);
  }

  // setup PET (Profile Every Thread), start sampler
  u64 tick = kperf_ns_to_ticks(sample_period * 1000000000ul);
  if ((ret = kperf_timer_period_set(actionid, tick))) {
    printf("Failed set timer period: %d.\n", ret);
  }
  if ((ret = kperf_timer_action_set(actionid, timerid))) {
    printf("Failed set timer action: %d.\n", ret);
  }
  if ((ret = kperf_timer_pet_set(timerid))) {
    printf("Failed set timer PET: %d.\n", ret);
  }
  if ((ret = kperf_lightweight_pet_set(1))) {
    printf("Failed set lightweight PET: %d.\n", ret);
  }
  if ((ret = kperf_sample_set(1))) {
    printf("Failed start sample: %d.\n", ret);
  }

  // reset kdebug/ktrace
  if ((ret = kdebug_reset())) {
    printf("Failed reset kdebug: %d.\n", ret);
  }

  int nbufs = 1000000;
  if ((ret = kdebug_trace_setbuf(nbufs))) {
    printf("Failed setbuf: %d.\n", ret);
  }
  if ((ret = kdebug_reinit())) {
    printf("Failed init kdebug buffer: %d.\n", ret);
  }

  // set trace filter: only log PERF_KPC_DATA_THREAD
  kd_regtype kdr = {0};
  kdr.type = KDBG_VALCHECK;
  kdr.value1 = KDBG_EVENTID(DBG_PERF, PERF_KPC, PERF_KPC_DATA_THREAD);
  if ((ret = kdebug_setreg(&kdr))) {
    printf("Failed set kdebug filter: %d.\n", ret);
  }
  // start trace
  if ((ret = kdebug_trace_enable(1))) {
    printf("Failed enable kdebug trace: %d.\n", ret);
  }

  // sample and get buffers
  usize buf_capacity = nbufs * 2;
  kd_buf *buf_hdr = (kd_buf *)malloc(sizeof(kd_buf) * buf_capacity);
  kd_buf *buf_cur = buf_hdr;
  kd_buf *buf_end = buf_hdr + buf_capacity;

  while (buf_hdr) {
    // wait for more buffer
    usleep(2 * sample_period * 1000000);

    // expand local buffer for next read
    if (buf_end - buf_cur < nbufs) {
      usize new_capacity = buf_capacity * 2;
      kd_buf *new_buf =
          (kd_buf *)realloc(buf_hdr, sizeof(kd_buf) * new_capacity);
      if (!new_buf) {
        free(buf_hdr);
        buf_hdr = NULL;
        break;
      }
      buf_capacity = new_capacity;
      buf_cur = new_buf + (buf_cur - buf_hdr);
      buf_end = new_buf + (buf_end - buf_hdr);
      buf_hdr = new_buf;
    }

    // read trace buffer from kernel
    usize count = 0;
    kdebug_trace_read(buf_cur, sizeof(kd_buf) * nbufs, &count);
    for (kd_buf *buf = buf_cur, *end = buf_cur + count; buf < end; buf++) {
      u32 debugid = buf->debugid;
      u32 cls = KDBG_EXTRACT_CLASS(debugid);
      u32 subcls = KDBG_EXTRACT_SUBCLASS(debugid);
      u32 code = KDBG_EXTRACT_CODE(debugid);

      // keep only thread PMC data
      if (cls != DBG_PERF)
        continue;
      if (subcls != PERF_KPC)
        continue;
      if (code != PERF_KPC_DATA_THREAD)
        continue;
      memmove(buf_cur, buf, sizeof(kd_buf));
      buf_cur++;
    }

    int status = 0;
    if (waitpid(cmd_pid, &status, WNOHANG) != 0)
      break;
  }

  // stop tracing
  kdebug_trace_enable(0);
  kdebug_reset();
  kperf_sample_set(0);
  kperf_lightweight_pet_set(0);

  // stop counting
  kpc_set_counting(0);
  kpc_set_thread_counting(0);
  kpc_force_all_ctrs_set(0);

  // aggregate thread PMC data
  if (!buf_hdr) {
    printf("Failed to allocate memory for trace log.\n");
    return 1;
  }
  if (buf_cur - buf_hdr == 0) {
    printf("No thread PMC data collected.\n");
    return 1;
  }

  typedef struct {
    u32 tid;
    u64 timestamp_0;
    u64 timestamp_1;
    u64 counters_0[KPC_MAX_COUNTERS];
    u64 counters_1[KPC_MAX_COUNTERS];
  } kpc_thread_data;

  usize thread_capacity = 16;
  usize thread_count = 0;
  kpc_thread_data *thread_data =
      (kpc_thread_data *)malloc(thread_capacity * sizeof(kpc_thread_data));
  if (!thread_data) {
    printf("Failed to allocate memory for aggregate log.\n");
    return 1;
  }
  for (kd_buf *buf = buf_hdr; buf < buf_cur; buf++) {
    u32 func = buf->debugid & KDBG_FUNC_MASK;
    if (func != DBG_FUNC_START)
      continue;
    u32 tid = (u32)buf->arg5;
    if (!tid)
      continue;

    // read one counter log
    u32 ci = 0;
    u64 counters[KPC_MAX_COUNTERS];
    counters[ci++] = buf->arg1;
    counters[ci++] = buf->arg2;
    counters[ci++] = buf->arg3;
    counters[ci++] = buf->arg4;
    if (ci < counter_count) {
      // counter count larger than 4
      // values are split into multiple buffer entities
      for (kd_buf *buf2 = buf + 1; buf2 < buf_cur; buf2++) {
        u32 tid2 = (u32)buf2->arg5;
        if (tid2 != tid)
          break;
        u32 func2 = buf2->debugid & KDBG_FUNC_MASK;
        if (func2 == DBG_FUNC_START)
          break;
        if (ci < counter_count)
          counters[ci++] = buf2->arg1;
        if (ci < counter_count)
          counters[ci++] = buf2->arg2;
        if (ci < counter_count)
          counters[ci++] = buf2->arg3;
        if (ci < counter_count)
          counters[ci++] = buf2->arg4;
        if (ci == counter_count)
          break;
      }
    }
    if (ci != counter_count)
      continue; // not enough counters, maybe truncated

    // add to thread data
    kpc_thread_data *data = NULL;
    for (usize i = 0; i < thread_count; i++) {
      if (thread_data[i].tid == tid) {
        data = thread_data + i;
        break;
      }
    }
    if (!data) {
      if (thread_capacity == thread_count) {
        thread_capacity *= 2;
        kpc_thread_data *new_data = (kpc_thread_data *)realloc(
            thread_data, thread_capacity * sizeof(kpc_thread_data));
        if (!new_data) {
          printf("Failed to allocate memory for aggregate log.\n");
          return 1;
        }
        thread_data = new_data;
      }
      data = thread_data + thread_count;
      thread_count++;
      memset(data, 0, sizeof(kpc_thread_data));
      data->tid = tid;
    }
    if (data->timestamp_0 == 0) {
      data->timestamp_0 = buf->timestamp;
      memcpy(data->counters_0, counters, counter_count * sizeof(u64));
    } else {
      data->timestamp_1 = buf->timestamp;
      memcpy(data->counters_1, counters, counter_count * sizeof(u64));
    }
  }

  printf("\n");
  printf("  Perfomance counters stats for '");
  for (size_t i = 0; i < exec_argc; i++) {
    if (i > 0) {
      printf(" ");
    }
    printf("%s", exec_argv[i]);
  }
  printf("'\n");
  printf("\n");

  // Sort thread_data array by thread id
  for (usize i = 0; i < thread_count - 1; i++) {
    for (usize j = i + 1; j < thread_count; j++) {
      if (thread_data[i].tid > thread_data[j].tid) {
        kpc_thread_data temp = thread_data[i];
        thread_data[i] = thread_data[j];
        thread_data[j] = temp;
      }
    }
  }

  for (usize i = 0; i < thread_count; i++) {
    kpc_thread_data *data = thread_data + i;
    if (!data->timestamp_0 || !data->timestamp_1)
      continue;

    u64 counters_one[KPC_MAX_COUNTERS] = {0};
    for (usize c = 0; c < counter_count; c++) {
      counters_one[c] += data->counters_1[c] - data->counters_0[c];
    }

    printf("  thread: %u\n\n", data->tid);
    for (usize i = 0; i < ev_count; i++) {
      const event_alias *alias = profile_events + i;
      u64 val = counters_one[counter_map[i]];
      printf(" %12llu %-14s # %-30s\n", val, alias->alias, ev_arr[i]->name);
    }
    printf(" %12.6f seconds elapsed\n",
           kperf_ticks_to_ns(data->timestamp_1 - data->timestamp_0) /
               1000000000.0);

    // Calculating IPC. Here we're relying on the fact that cycles and instructions
    // are the firt two metrics defined in the profile_events array
    u64 cycles = counters_one[counter_map[0]];
    u64 instructions = counters_one[counter_map[1]];
    printf(" %12.2f IPC\n", (f64)instructions / cycles);

    printf("\n");
  }

  // TODO: free memory
  return 0;
}
