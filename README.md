Sistem Pakar Identifikasi Jenis Bunga

Proyek ini merupakan aplikasi berbasis web yang dapat mengidentifikasi jenis bunga dari gambar menggunakan metode **Convolutional Neural Network (CNN)** dengan framework **Flask**.

 Fitur

- Upload gambar bunga
- Identifikasi otomatis menggunakan model CNN
- Tampilkan hasil prediksi dan tingkat kepercayaan
- Antarmuka sederhana dengan efek kaca dan background bunga

#Teknologi

- Python
- TensorFlow / Keras
- Flask
- HTML + CSS (Glassmorphism UI)
- Git

🗂️ Struktur Folder
├── Dataset/ # Folder dataset bunga (tidak disertakan di GitHub)
├── static/
│ └── uploads/ # Folder menyimpan gambar yang diupload user
├── templates/
│ ├── dashboard.html # Tampilan dashboard untuk upload
│ └── result.html # Tampilan hasil identifikasi
├── model_bunga.h5 # Model hasil training CNN
├── label_map.json # Mapping label indeks ke nama bunga
├── app.py # Aplikasi utama Flask
├── training_model.py # Script pelatihan model CNN
└── .gitignore # Mengabaikan dataset saat push


## 📦 Cara Menjalankan

1. Clone repositori ini:
   ```bash
   git clone https://github.com/NurdiansyahLg/identifikasi-bunga.git
   cd identifikasi-bunga
2. Aktifkan virtual environment (opsional):
   python -m venv venv
   venv\Scripts\activate  # Windows
4. Install dependensi
   pip install -r requirements.txt
5. Jalankan Aplikasi
   python app.py

📌 Catatan
Folder Dataset/ tidak disertakan di GitHub karena ukurannya besar dan hanya digunakan saat training model.
   
