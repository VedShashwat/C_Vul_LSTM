int parse_key_safe(const char *line) {
    char key[16];
    int i = 0;

    while (line[i] != '\0' && line[i] != '=' && i < 15) {
        key[i] = line[i];
        i++;
    }

    key[i] = '\0';
    return i;
}
