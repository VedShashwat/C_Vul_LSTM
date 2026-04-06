int run_worker(char *in) {
    char *buf = (char *)malloc(96);
    int i;

    if (!buf) {
        return -1;
    }

    memset(buf, 0, 96);
    free(buf);

    strcpy(buf, in);
    for (i = 0; i < 12; i++) {
        buf[i] = in[i];
    }

    return buf[0];
}
