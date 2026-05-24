import os
import re
import pandas as pd
from collections import Counter

BASE_DIR = "cleaned"

# =========================
# FIXED FEATURE SCHEMA
# =========================
COLUMNS = [
    # syscall features (binary presence style frequency)
    "accept","access","arch_prctl","bind","brk","chdir","chmod",
    "clock_nanosleep","clone","close","close_range","connect",
    "copy_file_range","dup2","epoll_create1","epoll_ctl","epoll_pwait",
    "execve","exit","exit_group","faccessat2","fadvise64","fchdir","fchmod",
    "fcntl","flock","fork","fstat","fsync","ftruncate","futex","getcwd",
    "getdents64","getegid","geteuid","getgid","getgroups","getpid","getppid",
    "getrandom","getrlimit","getsockname","getsockopt","gettid","getuid",
    "ioctl","kill","listen","lseek","madvise","mkdir","mmap","mprotect",
    "munmap","nanosleep","newfstatat","open","openat","pipe","pipe2","poll",
    "prctl","pread64","prlimit64","read","readlink","readlinkat","recvfrom",
    "rt_sigaction","rt_sigprocmask","rt_sigreturn","select","sendto",
    "set_robust_list","set_tid_address","setsid","setsockopt","setuid",
    "sigaltstack","socket","splice","stat","statfs","statx","tgkill",
    "time","times","umask","uname","unlink","unlinkat","vfork","wait4","write",

    # engineered behavioral features
    "network_ratio",
    "file_ratio",
    "memory_ratio",
    "process_ratio",
    "syscall_diversity",
    "total_syscalls",

    "label"
]

# =========================
# LABELING
# =========================
def get_label(folder):
    f = folder.lower()
    if "benign" in f:
        return 0
    elif "mirai" in f or "gafgyt" in f:
        return 1
    return None

# =========================
# ROBUST SYSCALL PARSER
# =========================
def extract_syscalls(path):
    syscalls = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            # extract syscall name safely
            match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]+)\b", line)
            if match:
                syscalls.append(match.group(1))

    return syscalls

# =========================
# FEATURE BUILDER (IMPROVED)
# =========================
def build_features(syscalls):
    count = Counter(syscalls)
    total = sum(count.values()) if count else 1

    features = {}

    # -------------------------
    # syscall frequency features (normalized)
    # -------------------------
    for col in COLUMNS:
        if col in ["label", "network_ratio", "file_ratio",
                   "memory_ratio", "process_ratio",
                   "syscall_diversity", "total_syscalls"]:
            continue

        # normalized frequency
        features[col] = count[col] / total if total > 0 else 0.0

    # -------------------------
    # behavioral ratios (IMPORTANT)
    # -------------------------
    network_calls = {"socket","connect","sendto","recvfrom","bind","listen","accept","getsockopt","setsockopt"}
    file_calls = {"open","openat","read","write","close","fstat","newfstatat","lseek","unlink","unlinkat"}
    memory_calls = {"mmap","mprotect","brk","munmap"}
    process_calls = {"fork","clone","vfork","execve","wait4","kill","tgkill"}

    features["network_ratio"] = sum(count[s] for s in network_calls) / total
    features["file_ratio"] = sum(count[s] for s in file_calls) / total
    features["memory_ratio"] = sum(count[s] for s in memory_calls) / total
    features["process_ratio"] = sum(count[s] for s in process_calls) / total

    # -------------------------
    # global behavior features
    # -------------------------
    features["total_syscalls"] = total
    features["syscall_diversity"] = len(count)

    return features

# =========================
# DATA LOADER (FIXED STRUCTURE)
# =========================
def load_dataset():
    data = []

    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        label = get_label(folder)
        if label is None:
            continue

        print(f"[+] Processing: {folder}")

        for file in os.listdir(folder_path):
            if not file.endswith(".txt"):
                continue

            path = os.path.join(folder_path, file)

            syscalls = extract_syscalls(path)
            features = build_features(syscalls)

            features["label"] = label
            data.append(features)

    df = pd.DataFrame(data)

    # enforce column order strictly
    df = df[[c for c in COLUMNS if c in df.columns]]

    return df

# =========================
# RUN
# =========================
if __name__ == "__main__":
    df = load_dataset()

    df.to_csv("dataset.csv", index=False)

    print("\nDONE")
    print("Shape:", df.shape)
    print(df["label"].value_counts())
