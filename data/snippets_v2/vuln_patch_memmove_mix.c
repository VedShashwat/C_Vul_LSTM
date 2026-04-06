int apply_patch(unsigned int count, char *base, char *delta) {
    unsigned int bytes = count * 32;
    char *dst = (char *)malloc(bytes);
    unsigned int i;

    if (!dst) {
        return -1;
    }

    for (i = 0; i < count; i++) {
        memmove(dst + i * 32, base + i * 32, 32);
        memcpy(dst + i * 32, delta + i * 32, 32);
    }

    free(dst);
    return 0;
}
