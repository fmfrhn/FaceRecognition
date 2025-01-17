import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt  # Tambahkan import ini di bagian atas program

# Folder dataset untuk menyimpan data wajah
DATASET_DIR = './dataset'

# Membuat folder dataset jika belum ada
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Fungsi untuk menampilkan histogram dataset
def show_dataset_histogram():
    if not os.listdir(DATASET_DIR):
        messagebox.showwarning("Dataset Kosong", "Tidak ada data dalam dataset untuk ditampilkan.")
        return

    histogram = np.zeros(256, dtype=int)  # Inisialisasi array histogram

    for file_name in os.listdir(DATASET_DIR):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(DATASET_DIR, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            histogram += hist.flatten().astype(int)  # Gabungkan histogram setiap gambar

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.title("Histogram Dataset")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(histogram, color='blue')
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

# Fungsi untuk merekam wajah
def record_face():
    name = simpledialog.askstring("Input", "Masukkan Nama:")
    nim = simpledialog.askstring("Input", "Masukkan NIM:")

    if not name or not nim:
        messagebox.showerror("Input Error", "Nama dan NIM harus diisi.")
        return
    
    # Membuka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Kamera Error", "Tidak dapat membuka kamera.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    while count < 20:  # Mengambil 10 gambar wajah
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Kamera Error", "Tidak dapat membaca frame dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            file_path = os.path.join(DATASET_DIR, f"{name}_{nim}_{count}.jpg")
            cv2.imwrite(file_path, face)  # Simpan gambar wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Rekam Wajah", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Latih model setelah wajah direkam
    face_recognizer, label_dict = train_model()
    if face_recognizer is not None:
        messagebox.showinfo("Training Selesai", "Model berhasil dilatih dengan dataset yang ada.")
    else:
        messagebox.showerror("Training Gagal", "Gagal melatih model. Tambahkan lebih banyak gambar.")
        
# Fungsi untuk melatih model LBPH untuk pengenalan wajah
def train_model():
    images = []
    labels = []
    label_dict = {}
    current_id = 0
    
    for file_name in os.listdir(DATASET_DIR):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(DATASET_DIR, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            name_nim = file_name.split('_')[0] + "_" + file_name.split('_')[1]  # Nama_NIM
            
            if name_nim not in label_dict:
                label_dict[name_nim] = current_id
                current_id += 1
            
            images.append(np.array(img))
            labels.append(label_dict[name_nim])
    
    labels = np.array(labels)
    
    # Validasi: Pastikan ada lebih dari 1 gambar dalam dataset
    if len(images) < 2:
        return None, None

    # Membuat model LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(images, labels)
    
    # Simpan model yang sudah dilatih
    face_recognizer.save('face_recognition_model.yml')
    return face_recognizer, label_dict

# Fungsi untuk scan wajah dan mengenali wajah
def scan_face():
    # Cek apakah model sudah dilatih dan disimpan
    if not os.path.exists('face_recognition_model.yml'):
        messagebox.showerror("Model Error", "Model belum dilatih. Rekam wajah terlebih dahulu.")
        return
    
    # Muat model dan dictionary label
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_recognition_model.yml')
    
    # Muat dictionary label
    label_dict = {}
    current_id = 0
    for file_name in os.listdir(DATASET_DIR):
        if file_name.endswith(".jpg"):
            name_nim = file_name.split('_')[0] + "_" + file_name.split('_')[1]
            if name_nim not in label_dict:
                label_dict[name_nim] = current_id
                current_id += 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Kamera Error", "Tidak dapat membuka kamera.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Kamera Error", "Tidak dapat membaca frame dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            
            label_id, confidence = face_recognizer.predict(face)
            
            # Jika confidence rendah, artinya prediksi kurang tepat
            if confidence < 100:  # Confidence range dari 0-100, makin rendah makin cocok
                name_nim = list(label_dict.keys())[list(label_dict.values()).index(label_id)]
                cv2.putText(frame, f"{name_nim} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                # Jika wajah tidak cocok
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        cv2.imshow("Scan Wajah", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk membuat GUI
def create_gui():
    root = tk.Tk()
    root.title("Face Recognition")
    root.geometry("400x400")
    root.configure(bg="#f0f0f0")

    # Style for buttons
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 14), padding=10)

    # Title label
    title_label = tk.Label(root, text="Face Recognition System", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
    title_label.pack(pady=20)

    # Rekam Wajah button
    record_button = ttk.Button(root, text="Rekam Wajah", command=record_face)
    record_button.pack(pady=10)
    
    # Scan Wajah button
    scan_button = ttk.Button(root, text="Scan Wajah", command=scan_face)
    scan_button.pack(pady=10)

    # Tampilkan Histogram Dataset button
    histogram_button = ttk.Button(root, text="Tampilkan Histogram Dataset", command=show_dataset_histogram)
    histogram_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
