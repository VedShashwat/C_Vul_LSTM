int to_u8(int x) {
    if (x < 0 || x > 255) {
        return -1;
    }
    return x;
}
