# Implementasi-Website-Knowledge-Distillation
Aplikasi ini merupakan implementasi sistem **klasifikasi penyakit daun tomat**.
Model yang digunakan adalah **EfficientNetB0** hasil **Knowledge Distillation**
dari model Teacher EfficientNetB5, kemudian dideploy sebagai aplikasi web
menggunakan **Flask** dan dipublikasikan melalui **Hugging Face Spaces**.

---

## 🔗 Demo & Repository

- **Repository GitHub Model Development**  
  https://github.com/RolandSitompul750/Knowledge-Distilation

- **Aplikasi Web (Public Deployment)**  
  https://huggingface.co/spaces/rolandsitompul750/Deteksi_Penyakit_Daun_Tomat

---

## 📌 Fitur Aplikasi

- Upload foto daun tomat (.jpg, .jpeg, .png)
- Klasifikasi otomatis ke dalam **10 kelas penyakit daun tomat**
- Menampilkan:
  - Hasil diagnosis penyakit
  - Nilai confidence model
  - Top-3 prediksi kelas tertinggi
  - Deskripsi singkat penyakit yang terdeteksi
- Penolakan otomatis gambar bukan daun tomat menggunakan **confidence threshold**

---

## 🧠 Model & Metodologi

- **Arsitektur**: EfficientNetB0 (Student)
- **Metode**: Knowledge Distillation dari EfficientNetB5 (Teacher)
- **Parameter Distilasi**:
  - Temperature (T) = 3
  - Alpha (α) = 0.1

Model dikembangkan dan dilatih secara terpisah di repository model,
kemudian digunakan langsung dalam format `.keras` untuk keperluan deployment.

---

## 📝 Catatan Penting

- Output layer model menggunakan `activation=None` (logits mentah) karena
  proses Knowledge Distillation membutuhkan logits untuk dibagi Temperature
  saat training. Konversi ke probabilitas dilakukan secara eksplisit
  menggunakan `tf.nn.softmax()` di sisi aplikasi sebelum ditampilkan ke pengguna.
- Confidence threshold sebesar **75%** digunakan untuk menolak gambar
  yang bukan merupakan daun tomat.
- Aplikasi ini ditujukan untuk keperluan penelitian dan demonstrasi sistem
  klasifikasi penyakit daun tomat.

---

**Author**  
Roland Sitompul
