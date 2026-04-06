int write_temp_record(const char *payload, unsigned int payload_len) {
    char pattern[] = "/tmp/recordXXXXXX";
    int fd = mkstemp(pattern);

    if (fd < 0) {
        return -1;
    }

    if (write(fd, payload, payload_len) != (ssize_t)payload_len) {
        close(fd);
        unlink(pattern);
        return -1;
    }

    close(fd);
    unlink(pattern);
    return 0;
}
