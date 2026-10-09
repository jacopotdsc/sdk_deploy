#pragma once

#include <ament_index_cpp/get_package_prefix.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>

extern char **environ;

// Own the background MPC worker for the lifetime of rl_deploy.
class MPCProcess {
    pid_t pid_ = -1;
public:
    MPCProcess() = default;
    MPCProcess(const MPCProcess&) = delete;
    MPCProcess& operator=(const MPCProcess&) = delete;
    ~MPCProcess() { Stop(); }
    void Start() {
        if (pid_ > 0) return;
        const std::string script = ament_index_cpp::get_package_prefix("lite3_sdk_deploy")
            + "/lib/lite3_sdk_deploy/mpc_srbd";
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        posix_spawnattr_t attributes;
        posix_spawnattr_init(&attributes);
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attributes, 0);
        char *argv[] = {const_cast<char*>(script.c_str()), nullptr};
        const int error = posix_spawn(&pid_, script.c_str(), &actions, &attributes, argv, environ);
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attributes);
        if (error != 0) {
            pid_ = -1;
            std::cerr << "[MPC] Start failed: " << std::strerror(error) << ". The RL policy remains available.\n";
        } else {
            std::cout << "[MPC] Preparing in background; Z/X available, C / A selects controllers; mpx after MPC READY.\n";
        }
    }
    void Stop() {
        if (pid_ <= 0) return;
        const pid_t child = pid_;
        pid_ = -1;
        if (waitpid(child, nullptr, WNOHANG) == child) return;
        kill(-child, SIGTERM);
        for (int i = 0; i < 20; ++i) {
            const pid_t result = waitpid(child, nullptr, WNOHANG);
            if (result == child || (result < 0 && errno == ECHILD)) return;
            usleep(50000);
        }
        kill(-child, SIGKILL);
        while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
    }
};
