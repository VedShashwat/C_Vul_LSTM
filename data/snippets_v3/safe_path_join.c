int read_profile_safe(const char *user_id, char *dst, int cap) {
    char path[160];
    int i = 0;
    int fd;

    while (user_id[i] != '\0') {
        if (user_id[i] == '/' || (user_id[i] == '.' && user_id[i + 1] == '.')) {
            return -1;
        }
        i++;
    }

    snprintf(path, sizeof(path), "/srv/profiles/%s", user_id);
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    i = read(fd, dst, cap > 0 ? cap - 1 : 0);
    if (i >= 0) {
        dst[i] = '\0';
    }
    close(fd);
    return i;
}
