# Adam Ullmann
def read_ppm(filename):
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header != 'P3':
            raise ValueError("unsupported file format. Only P3 PPM is supported.")
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
    return width, height, max_val, pixels

def write_ppm(filename, width, height, max_val, pixels):
    with open(filename, 'w') as f:
        f.write("P3\n")
        f.write(f"{width} {height}\n")
        f.write(f"{max_val}\n")
        for i in range(0, len(pixels), 3):
            f.write(f"{pixels[i]} {pixels[i+1]} {pixels[i+2]}\n")

def compute_highlighted_differences(image1, image2):
    width1, height1, max_val1, pixels1 = image1
    width2, height2, max_val2, pixels2 = image2
    if width1 != width2 or height1 != height2:
        raise ValueError("images must have the same dimensions.")
    max_val = max(max_val1, max_val2)
    diff_pixels = []
    for i in range(0, len(pixels1), 3):
        if pixels1[i:i+3] == pixels2[i:i+3]: #identical 
            diff_pixels.extend([0, 0, 0])
        else:                               # different
            diff_pixels.extend([max_val, max_val, max_val])
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
        diff_image = compute_highlighted_differences(image1, image2)
        write_ppm(output_path, *diff_image)
        print(f"dif image written to {output_path}")
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    main()

