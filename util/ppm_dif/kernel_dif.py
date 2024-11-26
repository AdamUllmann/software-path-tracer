# Adam Ullmann
import numpy as np

def read_ppm(filename):
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header != 'P3':
            raise ValueError("unsupported file format. only P3 PPM is supported.")
        dimensions = f.readline().strip()
        while dimensions.startswith('#'):
            dimensions = f.readline().strip()
        width, height = map(int, dimensions.split())
        max_val = int(f.readline().strip())
        pixels = []
        for line in f:
            if line.startswith('#'):
                continue
            pixels.extend(map(int, line.split()))
    return width, height, max_val, np.array(pixels).reshape(height, width, 3)

def write_ppm(filename, width, height, max_val, pixels):
    with open(filename, 'w') as f:
        f.write("P3\n")
        f.write(f"{width} {height}\n")
        f.write(f"{max_val}\n")
        for row in pixels:
            for pixel in row:
                f.write(f"{pixel[0]} {pixel[1]} {pixel[2]} ")
            f.write("\n")

def compute_complex_differences(image1, image2, threshold=15):
    """
    Compares two PPM images using a 5x5 kernel, averages, and thresholds.
    outputs differences that exceed the threshold.
    """
    width1, height1, max_val1, pixels1 = image1
    width2, height2, max_val2, pixels2 = image2
    if width1 != width2 or height1 != height2:
        raise ValueError("images must have the same dimensions.")
    max_val = max(max_val1, max_val2)
    height, width, _ = pixels1.shape
    diff_pixels = np.zeros_like(pixels1)
    for y in range(height):
        for x in range(width):
            y_min, y_max = max(0, y - 2), min(height - 1, y + 2)
            x_min, x_max = max(0, x - 2), min(width - 1, x + 2)
            neighborhood1 = pixels1[y_min:y_max + 1, x_min:x_max + 1]
            neighborhood2 = pixels2[y_min:y_max + 1, x_min:x_max + 1]
            avg1 = np.mean(neighborhood1, axis=(0, 1))
            avg2 = np.mean(neighborhood2, axis=(0, 1))
            diff = np.linalg.norm(avg1 - avg2)
            if diff > threshold:
                diff_pixels[y, x] = [max_val, max_val, max_val]  # white
            else:
                diff_pixels[y, x] = [0, 0, 0]  # black
    return width1, height1, max_val, diff_pixels

def main():
    import sys
    if len(sys.argv) != 4:
        print("Usage: python diff_ppm.py <image1.ppm> <image2.ppm> <output.ppm>")
        return
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    output_path = sys.argv[3]
    try:
        image1 = read_ppm(image1_path)
        image2 = read_ppm(image2_path)
        diff_image = compute_complex_differences(image1, image2)
        write_ppm(output_path, *diff_image)
        print(f"difference image written to {output_path}")
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()

