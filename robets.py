import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Fungsi untuk menerapkan operator Roberts
def roberts_edge_detection(image):
    # Operator Roberts untuk gradien Gx dan Gy
    kernel_gx = np.array([[1, 0], [0, -1]], dtype=float)
    kernel_gy = np.array([[0, 1], [-1, 0]], dtype=float)
    
    # Menghitung gradien Gx dan Gy menggunakan konvolusi
    grad_x = convolve(image, kernel_gx)
    grad_y = convolve(image, kernel_gy)
    
    # Menghitung magnitude gradien
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return magnitude

# Fungsi utama
def main(image_path):
    # Membaca gambar menggunakan imageio
    image = imageio.imread(image_path)
    
    # Jika gambar berwarna (RGB), konversi ke grayscale
    if image.ndim == 3:
        image = np.mean(image, axis=2)  # Rata-rata RGB menjadi grayscale
    
    # Terapkan deteksi tepi menggunakan operator Roberts
    edges = roberts_edge_detection(image)
    
    # Menampilkan hasil
    plt.figure(figsize=(10, 5))
    
    # Gambar asli
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Gambar Asli')
    plt.axis('off')
    
    # Gambar hasil deteksi tepi
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Deteksi Tepi (Roberts)')
    plt.axis('off')
    
    plt.show()

# Masukkan path gambar
image_path = 'pcd9.jpeg'  # Ganti dengan path gambar Anda
main(image_path)
