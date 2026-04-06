int expand_rows(unsigned int rows, char *blob) {
    unsigned int bytes = rows * 128;
    char *tmp = (char *)malloc(bytes);
    unsigned int i;

    if (!tmp) {
        return -1;
    }

    for (i = 0; i < rows; i++) {
        memcpy(tmp + i * 128, blob + i * 128, 128);
    }

    free(tmp);
    return 0;
}
