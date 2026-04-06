unsigned int crc32_step(unsigned int crc, unsigned char b) {
    unsigned int i;
    crc ^= b;

    for (i = 0; i < 8; i++) {
        if (crc & 1U) {
            crc = (crc >> 1) ^ 0xEDB88320U;
        } else {
            crc >>= 1;
        }
    }

    return crc;
}
