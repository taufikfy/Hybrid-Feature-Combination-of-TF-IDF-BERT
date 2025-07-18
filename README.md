**Sistem Pencarian Dokumen Hibrida: Kombinasi TF-IDF dan BERT**
<img width="1920" height="1025" alt="Cuplikan layar 2025-07-17 075450" src="https://github.com/user-attachments/assets/ce248114-a8ad-4cdb-ac0a-29cb41c47c53" />
**Deskripsi Proyek**

<p>Proyek ini mengimplementasikan sebuah sistem Information Retrieval (IR) hibrida yang dirancang untuk meningkatkan akurasi pencarian dokumen dengan 
menggabungkan kekuatan representasi teks tradisional (TF-IDF) dan modern berbasis deep learning (BERT). Dalam era information overload, kemampuan untuk menemukan informasi yang relevan secara efisien menjadi sangat krusial. Sistem ini mengatasi keterbatasan pendekatan tunggal dengan memadukan relevansi leksikal dari TF-IDF dan pemahaman semantik kontekstual dari BERT.</p>	

**Metodologi Inti**

Sistem ini dibangun berdasarkan metodologi feature-level fusion yang diusulkan dalam penelitian "Hybrid Feature Combination of TF-IDF and BERT for Enhanced Information Retrieval Accuracy" oleh Aprilio et al. (2025).

- TF-IDF (Term Frequency-Inverse Document Frequency): Digunakan untuk menangkap pentingnya kata kunci dalam dokumen secara statistik. Vektor TF-IDF direduksi dimensinya menggunakan Truncated SVD.

- BERT (Bidirectional Encoder Representations from Transformers): Model all-MiniLM-L6-v2 dari Sentence Transformers digunakan untuk menghasilkan embedding kontekstual yang kaya secara semantik dari dokumen dan kueri.

- Feature Fusion: Vektor TF-IDF yang direduksi dan embedding BERT digabungkan secara berbobot (BERT 90%, TF-IDF 10%) untuk menciptakan representasi hibrida yang komprehensif.

- Klasifikasi Relevansi: Representasi hibrida ini kemudian digunakan sebagai masukan untuk Jaringan Neural Terhubung Penuh (FCNN) yang dilatih untuk mengklasifikasikan relevansi antara kueri dan dokumen.

**Fitur Aplikasi**

- Pencarian Dokumen: Memungkinkan pengguna memasukkan kueri dan mendapatkan daftar dokumen yang relevan, diurutkan berdasarkan skor relevansi yang diprediksi oleh model hibrida.

- Detail Dokumen: Menampilkan konten lengkap dari dokumen yang dipilih.

- Ringkasan AI: Menghasilkan ringkasan singkat dan padat dari dokumen yang dipilih menggunakan model bahasa besar (Gemini API).

- Tampilan Metrik Evaluasi: Menampilkan metrik performa model (Akurasi Pelatihan, Akurasi Pengujian, Presisi, Recall, MAP) yang dihitung selama pelatihan backend.

Persyaratan Sistem

- Python 3.8+

- Visual Studio Code (Direkomendasikan sebagai IDE)

- Koneksi Internet (Diperlukan untuk mengunduh model BERT dan memanggil Gemini API)

- File Dataset CISI: CISI.ALL, CISI.QRY, CISI.REL (dapat diunduh dari https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/)

Langkah-langkah Instalasi dan Menjalankan Proyek
1. Kloning Repositori
git clone <URL_REPO_ANDA>
cd hybrid_ir_project

2. Unduh Dataset CISI
Kunjungi https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/.

Unduh CISI.ALL, CISI.QRY, dan CISI.REL.

Tempatkan ketiga file ini langsung di dalam folder backend/ proyek Anda. Pastikan nama file persis sama (termasuk huruf besar/kecil) dan tidak ada ekstensi tambahan seperti .txt.

3. Siapkan Backend
Navigasi ke Direktori Backend:

cd backend

Buat dan Aktifkan Lingkungan Virtual (Direkomendasikan):

python -m venv venv
# Untuk Windows:
.\venv\Scripts\activate
# Untuk macOS/Linux:
source venv/bin/activate

Instal Dependensi Python:

pip install -r requirements.txt

Proses ini mungkin memakan waktu beberapa menit karena mengunduh pustaka seperti TensorFlow dan Sentence Transformers.

Jalankan Server Flask:

python app.py

Server akan berjalan di http://127.0.0.1:5000. Perhatikan output di terminal Anda.

4. Latih Model
Setelah server Flask berjalan, jika model belum dilatih, Anda akan melihat pesan di terminal yang meminta Anda untuk melatihnya.

Buka browser web Anda dan kunjungi:

http://127.0.0.1:5000/train_model

Perhatikan terminal Anda. Proses pelatihan ini akan memakan waktu yang signifikan (beberapa menit hingga puluhan menit) karena melibatkan pengunduhan model BERT dan pelatihan neural network pada dataset CISI. Setelah selesai, terminal akan menampilkan metrik evaluasi dan mengkonfirmasi bahwa model telah disimpan di folder model_artifacts/.

5. Jalankan Frontend
Navigasi ke Direktori Frontend:

cd ../frontend

Buka index.html di Browser:

Cukup klik dua kali file index.html di File Explorer/Finder Anda, atau seret file tersebut ke jendela browser.
