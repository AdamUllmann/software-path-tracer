typedef struct {
	double x, y, z;
} Vec;

Vec vec(double x, double y, double z) {
	Vec v = {x, y, z};
	return v;
}

Vec add(Vec a, Vec b) {
	return vec(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec sub(Vec a, Vec b) {
	return vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec mul(Vec a, double b) {
	return vec(a.x * b, a.y * b, a.z * b);
}

Vec mult(Vec a, Vec b) {
	return vec(a.x * b.x, a.y * b.y, a.z * b.z);
}

double dot(Vec a, Vec b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec norm(Vec a) {
	double mag = sqrt(dot(a, a));
	return mul(a, 1 / mag);
}

Vec cross(Vec a, Vec b) {
	return vec(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		  );
}
