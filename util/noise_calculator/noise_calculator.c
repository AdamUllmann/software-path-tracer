#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CRITICAL_VALUE 50.0

typedef struct {
    unsigned char r, g, b;
} Pixel;

double calculateDistance(Pixel a, Pixel b) {
    return sqrt(pow(a.r - b.r, 2) + pow(a.g - b.g, 2) + pow(a.b - b.b, 2));
}
int isPixelNoisy(Pixel *pixels, int x, int y, int width, int height, double criticalValue) {
    Pixel current = pixels[y * width + x];
    int noisy = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                Pixel neighbor = pixels[ny * width + nx];
                if (calculateDistance(current, neighbor) > criticalValue) {
                    noisy = 1;
                    break;
                }
            }
        }
        if (noisy) break;
    }
    return noisy;
}
int calculateNoiseLevel(Pixel *pixels, int width, int height, double criticalValue) {
    int noisePixels = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (isPixelNoisy(pixels, x, y, width, height, criticalValue)) {
                noisePixels++;
            }
        }
    }

    return noisePixels;
}
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image.ppm>\n", argv[0]);
        return 1;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("error opening file");
        return 1;
    }
    char format[3];
    int width, height, maxVal;
    if (fscanf(file, "%2s\n%d %d\n%d\n", format, &width, &height, &maxVal) != 4 || strcmp(format, "P3") != 0 || maxVal != 255) {
        fprintf(stderr, "invalid PPM file format. must be P3 and max val 255\n");
        fclose(file);
        return 1;
    }
    Pixel *pixels = malloc(width * height * sizeof(Pixel));
    if (!pixels) {
        fprintf(stderr, "memory allocation failed\n");
        fclose(file);
        return 1;
    }
    for (int i = 0; i < width * height; i++) {
        if (fscanf(file, "%hhu %hhu %hhu", &pixels[i].r, &pixels[i].g, &pixels[i].b) != 3) {
            fprintf(stderr, "error reading pixel data\n");
            free(pixels);
            fclose(file);
            return 1;
        }
    }
    fclose(file);
    int noisePixels = calculateNoiseLevel(pixels, width, height, CRITICAL_VALUE);
    double noisePercentage = (double)noisePixels / (width * height) * 100.0;
    printf("# of noise pixels: %d (%.2f%% of total pixels)\n", noisePixels, noisePercentage);
    free(pixels);
    return 0;
}

