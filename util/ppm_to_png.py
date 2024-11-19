from PIL import Image
import sys

def ppm_to_png(ppm_file, png_file):
    try:
        with open(ppm_file, 'r') as f:
            magic_number = f.readline().strip()
            if magic_number != "P3":
                raise ValueError("unsupported file format. expected P3 PPM.")
            dimensions_line = f.readline().strip()
            while dimensions_line.startswith("#"):
                dimensions_line = f.readline().strip()
            width, height = map(int, dimensions_line.split())
            max_color_value = int(f.readline().strip())
            if max_color_value != 255:
                raise ValueError("unsupported max color value. expected 255.")
            pixel_data = []
            for line in f:
                if line.strip() and not line.startswith("#"):
                    pixel_data.extend(map(int, line.split()))
            if len(pixel_data) != width * height * 3:
                raise ValueError("mismatch in pixel data and image dimensions.")
            pixels = [
                tuple(pixel_data[i:i+3])
                for i in range(0, len(pixel_data), 3)
            ]
            image = Image.new("RGB", (width, height))
            image.putdata(pixels)
            image.save(png_file)
            print(f"Converted {ppm_file} to {png_file} successfully.")
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 ppm_to_png.py <input.ppm> <output.png>")
    else:
        ppm_to_png(sys.argv[1], sys.argv[2])

