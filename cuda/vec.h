#ifndef VEC_H
#define VEC_H

typedef struct {
    double x, y, z;
} Vec;

__host__ __device__ Vec vec(double x, double y, double z) {
    Vec v = {x, y, z};
    return v;
}

__host__ __device__ Vec add(Vec a, Vec b) {
    return vec(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ Vec sub(Vec a, Vec b) {
    return vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vec mul(Vec a, double b) {
    return vec(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ Vec mult(Vec a, Vec b) {
    return vec(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ double dot(Vec a, Vec b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec norm(Vec a) {
    return mul(a, 1 / sqrt(dot(a, a)));
}

__host__ __device__ Vec cross(Vec a, Vec b) {
    return vec(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

#endif // VEC_H
