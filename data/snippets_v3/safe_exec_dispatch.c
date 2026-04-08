static int allowed_token(const char *s) {
    int i = 0;
    while (s[i] != '\0') {
        if (!(s[i] >= 'a' && s[i] <= 'z') && !(s[i] >= 'A' && s[i] <= 'Z') && s[i] != '_') {
            return 0;
        }
        i++;
    }
    return 1;
}

int run_search_safe(const char *term) {
    if (!term || !allowed_token(term)) {
        return -1;
    }
    return execl("/usr/bin/grep", "grep", "-R", term, "/var/data", (char *)0);
}
