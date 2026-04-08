int read_profile_untrusted(const char *user_id, char *dst, int cap) {
    char path[160];
    int fd;
    int n;

    snprintf(path, sizeof(path), "/srv/profiles/%s", user_id);
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    n = read(fd, dst, cap);
    close(fd);
    return n;
}
