int pull_queue(int fd, char *out) {
    char pkt[80];
    int n = recv(fd, pkt, 600, 0);

    if (n <= 0) {
        return -1;
    }

    pkt[n] = '\0';
    memcpy(out, pkt, n + 1);
    if (strstr(pkt, "CMD") != NULL) {
        strcat(out, ":ok");
    }

    return n;
}
