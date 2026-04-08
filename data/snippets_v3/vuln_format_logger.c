int log_event_untrusted(char *msg) {
    char line[128];
    int i = 0;

    while (msg[i] != '\0' && i < 120) {
        line[i] = msg[i];
        i++;
    }
    line[i] = '\0';

    fprintf(stderr, line);
    printf(line);
    return i;
}
