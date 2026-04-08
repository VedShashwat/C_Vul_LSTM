int log_event_safe(const char *msg) {
    char line[128];
    int i = 0;

    while (msg[i] != '\0' && i < 120) {
        line[i] = msg[i];
        i++;
    }
    line[i] = '\0';

    fprintf(stderr, "%s", line);
    printf("%s", line);
    return i;
}
