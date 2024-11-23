#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "vec.h"

#define PI 3.14159265358979323846
#define clamp(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))
#define toInt(x) ((int)(pow(clamp(x), 1 / 2.2) * 255 + 0.5))
#define RAND() (drand48())
#define FOV 0.5135
#define MAX_BOUNCES 5

typedef struct {
	Vec origin;
    Vec direction;
} Ray;

typedef struct {
	Vec center;
	double radius;
	Vec color, emission;
	int material; // 0 is diffuse, 1 is reflective, 2 is refractive
} Sphere;

/*Sphere spheres[] = {
	{{0, -10004, -20}, 10000, {0.2, 0.2, 0.2}, {0, 0, 0}, 0}, // floor
	{{0, 0, -20}, 4, {1, 0.32, 0.36}, {0, 0, 0}, 0},          // rough Ball
	{{5, -1, -15}, 2, {0.9, 0.76, 0.46}, {0, 0, 0}, 1},       // reflective Ball
	{{-5, -1, -15}, 2, {0.7, 0.7, 1.0}, {0, 0, 0}, 2},        // glass Ball
	{{0, 20, -30}, 3, {0, 0, 0}, {3, 3, 3}, 0},               // light
};*/

Sphere spheres[] = {
    {{-10010, 0, -20}, 10000, {0.1, 0.9, 0.1}, {0, 0, 0}, 0},  // left Wall
    {{10010, 0, -20}, 10000, {0.9, 0.1, 0.1}, {0, 0, 0}, 0},   // right Wall
    {{0, -10004, -20}, 10000, {0.2, 0.2, 0.2}, {0, 0, 0}, 0},  // floor
    {{0, 0, -10040}, 10000, {0.75, 0.75, 0.75}, {0, 0, 0}, 0}, // back Wall
	{{0, 0, 10010}, 10000, {0.75, 0.75, 0.75}, {0, 0, 0}, 0},  // front wall
    {{-5, -2, -15}, 2, {0.7, 0.7, 1.0}, {0, 0, 0}, 0},         // diffuse Ball (left)
    {{5, -2, -15}, 2, {0.9, 0.76, 0.46}, {0, 0, 0}, 2},        // refractive Ball (right)
    {{0, -2, -20}, 2, {0.9, 0.9, 0.9}, {0, 0, 0}, 1},          // reflective Ball (center)
    {{0, 20, -30}, 4, {0, 0, 0}, {3, 3, 3}, 0},                // light
};

int sphere_count = sizeof(spheres) / sizeof(Sphere);
int intersect(Ray r, double *t, int *id) {
	double inf = 1e20;
	*t = inf;
	for (int i = 0; i < sphere_count; i++) {
		Sphere s = spheres[i];
		Vec oc = sub(r.origin, s.center);
		double b = dot(oc, r.direction);
		double c = dot(oc, oc) - s.radius * s.radius;
		double disc = b * b - c;
		if (disc > 0) {
			double d = sqrt(disc);
			double t0 = -b - d, t1 = -b + d;
			if (t0 > 1e-4 && t0 < *t) {
				*t = t0;
				*id = i;
			}
			if (t1 > 1e-4 && t1 < *t) {
				*t = t1;
				*id = i;
			}
		}
	}
	return *t < inf;
}

Vec trace(Ray r, int depth) {
	double t;
	int id = 0;
	if (!intersect(r, &t, &id)) return vec(0, 0, 0); // background color
	Sphere obj = spheres[id];
	Vec hit = add(r.origin, mul(r.direction, t));
	Vec normal = norm(sub(hit, obj.center));
	Vec nl = dot(normal, r.direction) < 0 ? normal : mul(normal, -1);
	Vec col = obj.color;
	double p = fmax(col.x, fmax(col.y, col.z));
	if (++depth > MAX_BOUNCES || !p) return obj.emission;
	if (obj.material == 0) { // diffuse
		double r1 = 2 * PI * RAND();
		double r2 = RAND(), r2s = sqrt(r2);
		Vec w = nl, u = norm(cross((fabs(w.x) > 0.1 ? vec(0, 1, 0) : vec(1, 0, 0)), w));
		Vec v = cross(w, u);
		Vec d = norm(add(add(mul(u, cos(r1) * r2s), mul(v, sin(r1) * r2s)), mul(w, sqrt(1 - r2))));
		return add(obj.emission, mult(col, trace((Ray){hit, d}, depth)));
	} else if (obj.material == 1) { // reflective
		Vec refl = sub(r.direction, mul(normal, 2 * dot(normal, r.direction)));
		return add(obj.emission, mult(col, trace((Ray){hit, refl}, depth)));
	} 
	else if (obj.material == 2) { // refractive
		Vec refl = sub(r.direction, mul(normal, 2 * dot(normal, r.direction))); // reflection
		int into = dot(normal, nl) > 0; // is the ray is entering the object? 0 of no, 1 if yes
		double nc = 1.0;                 // refractive index of air
		double nt = 1.5;                 // refractive index of object
		double nnt = into ? nc / nt : nt / nc;
		double ddn = dot(r.direction, nl);
		double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
		if (cos2t < 0) { // total internal reflection
			return add(obj.emission, mult(trace((Ray){hit, refl}, depth), col));
		}
		Vec tdir = norm(add(mul(r.direction, nnt), mul(normal, (into ? -1 : 1) * (nnt * ddn + sqrt(cos2t)))));
		double a = nt - nc, b = nt + nc;
		double R0 = (a * a) / (b * b);
		double c = 1 - (into ? -ddn : dot(tdir, normal));
		double Re = R0 + (1 - R0) * c * c * c * c * c; // reflectance
		double Tr = 1 - Re;                           // transmittance
		double P = 0.25 + 0.5 * Re;                   // reflection probability
		Vec refl_color = mult(trace((Ray){hit, refl}, depth), vec(Re, Re, Re));
		Vec refr_color = mult(trace((Ray){hit, tdir}, depth), vec(Tr, Tr, Tr));
		return add(obj.emission, 
				depth > 2 ? (RAND() < P 
					? refl_color 
					: refr_color) 
				: add(refl_color, refr_color));
	}
	return vec(0, 0, 0);
}

int main() {
	int width = 1600, height = 900, samples = 1;
	Vec camera = vec(0, 0, 10);
	Vec cx = vec(width * FOV / height, 0, 0); // aspect ratio
	Vec cy = norm(cross(cx, vec(0, 0, -1)));
	cy = mul(cy, FOV); // fov scaling
	Ray ray = {camera, {0, 0, -1}};
	FILE *f = fopen("output.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	Vec *image = (Vec *)calloc(width * height, sizeof(Vec));
	for (int y = 0; y < height; y++) {
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samples, 100. * y / (height - 1));
		for (int x = 0; x < width; x++) {
			Vec color = vec(0, 0, 0);
			for (int sy = 0; sy < 2; sy++) {
				for (int sx = 0; sx < 2; sx++) {
					for (int s = 0; s < samples; s++) {
						double r1 = 2 * RAND(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double r2 = 2 * RAND(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						Vec d = norm(add(add(mul(cx, ((sx + 0.5 + dx) / 2 + x) / width - 0.5),
										mul(cy, ((sy + 0.5 + dy) / 2 + y) / height - 0.5)),
									ray.direction));
						color = add(color, trace((Ray){camera, d}, 0));
					}
				}
			}
			image[y * width + x] = mul(color, 1. / (samples * 4));
		}
	}
	for (int y = height - 1; y >= 0; y--) { // flip & write to a file
		for (int x = 0; x < width; x++) {
			Vec pixel = image[y * width + x];
			fprintf(f, "%d %d %d ", toInt(pixel.x), toInt(pixel.y), toInt(pixel.z));
		}
	}
	fclose(f);
	free(image);
	return 0;
}
