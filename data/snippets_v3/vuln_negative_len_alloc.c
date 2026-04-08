int import_blob_untrusted(const char *size_txt, const char *src) {
    int len = atoi(size_txt);
    char *buf = (char *)malloc((size_t)len);

    if (!buf) {
        return -1;
    }

    memcpy(buf, src, 512);
    free(buf);
    return 0;
}
