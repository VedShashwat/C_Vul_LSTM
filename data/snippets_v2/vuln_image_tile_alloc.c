int decode_tile(unsigned int w, unsigned int h, char *src) {
    unsigned int pixels = w * h;
    unsigned int bytes = pixels * 4;
    char *dst = (char *)malloc(bytes);
    unsigned int i;

    if (!dst) {
        return -1;
    }

    for (i = 0; i < pixels; i++) {
        memcpy(dst + i * 4, src + i * 4, 4);
    }

    free(dst);
    return 0;
}
