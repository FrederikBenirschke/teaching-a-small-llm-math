// Log WHO sends SIGTERM/SIGINT/SIGHUP to a process, then die with that signal as before.
// macOS has no sigwaitinfo and Python handlers never see siginfo, so inject this instead:
//   cc -dynamiclib -o tools/sigwho.dylib tools/sigwho.c
//   DYLD_INSERT_LIBRARIES=tools/sigwho.dylib .venv/bin/python train_grpo.py ...
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void on_sig(int sig, siginfo_t *si, void *ctx) {
    char buf[160];
    int n = snprintf(buf, sizeof buf, "\n[sigwho] t=%ld got signal %d from pid %d uid %d; sender:\n",
                     (long)time(NULL), sig, (int)si->si_pid, (int)si->si_uid);
    write(2, buf, n);
    char pid[16];
    snprintf(pid, sizeof pid, "%d", (int)si->si_pid);
    pid_t c = fork();
    if (c == 0) {
        dup2(2, 1);
        char *env[] = {NULL};  // no DYLD_INSERT_LIBRARIES into ps
        execle("/bin/ps", "ps", "-o", "pid=,ppid=,user=,lstart=,command=", "-p", pid, (char *)NULL, env);
        _exit(1);
    }
    if (c > 0) waitpid(c, NULL, 0);
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_sig;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);
}
