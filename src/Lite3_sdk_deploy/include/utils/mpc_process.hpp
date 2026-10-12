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
#include <vector>
#include <stdexcept>

extern char **environ;

// Own one ROS child process and its process group.
class MPCProcess {
    pid_t pid_ = -1;
public:
    MPCProcess() = default;
    MPCProcess(const MPCProcess&) = delete;
    MPCProcess& operator=(const MPCProcess&) = delete;
    ~MPCProcess() { Stop(); }
    void Start(const std::string& executable, const std::vector<std::string>& arguments = {}) {
        if (pid_ > 0) return;
        const std::string script = ament_index_cpp::get_package_prefix("lite3_sdk_deploy")
            + "/lib/lite3_sdk_deploy/" + executable;
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        posix_spawnattr_t attributes;
        posix_spawnattr_init(&attributes);
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attributes, 0);
        std::vector<char*> argv{const_cast<char*>(script.c_str())};
        for (const auto& argument : arguments) argv.push_back(const_cast<char*>(argument.c_str()));
        argv.push_back(nullptr);
        const int error = posix_spawn(&pid_, script.c_str(), &actions, &attributes, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attributes);
        if (error != 0) {
            pid_ = -1;
            throw std::runtime_error("Cannot start " + executable + ": " + std::strerror(error));
        }

    }
    void Stop() {
        if (pid_ <= 0) return;
        const pid_t child = pid_;
        pid_ = -1;
        const pid_t initial = waitpid(child, nullptr, WNOHANG);
        if (initial == child || (initial < 0 && errno == ECHILD)) return;
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
