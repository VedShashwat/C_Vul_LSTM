int import_blob_safe(const char *size_txt, const char *src) {
    long len = strtol(size_txt, (char **)0, 10);
    char *buf;

    if (len <= 0 || len > 512) {
        return -1;
    }

    buf = (char *)malloc((size_t)len);
    if (!buf) {
        return -1;
    }

    memcpy(buf, src, (size_t)len);
    free(buf);
    return 0;
}
