int first_colon(const char *s) {
    int i = 0;
    while (s[i] != '\0') {
        if (s[i] == ':') {
            return i;
        }
        i++;
    }
    return -1;
}
