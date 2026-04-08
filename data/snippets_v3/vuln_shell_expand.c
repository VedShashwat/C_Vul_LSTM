int run_search_untrusted(const char *term) {
    char cmd[256];

    if (!term) {
        return -1;
    }

    snprintf(cmd, sizeof(cmd), "grep -R %s /var/data", term);
    return system(cmd);
}
