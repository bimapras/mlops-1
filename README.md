# Submission 1: Email Spam Detection 
Nama:Bima Prastyaji

Username dicoding: Bima Prastyaji

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Spam Dataset](https://www.kaggle.com/datasets/prishasawhney/email-classification-ham-spam) |
| Masalah | Email merupakan salah satu sarana komunikasi yang sering digunakan pada saat ini. Namun, keberadaan email juga sering dimanfaatkan untuk kegiatan yang tidak diinginkan seperti penyebaran ransowmware, penipuan, komersial, dall. Hal ini menyebabkan banyak pengguna yang sulit membedakan email spam dan email ham. |
| Solusi machine learning | Oleh karena itu, untuk membantu pengguna dalam mendeteksi sebuah email, maka dibutuhkan suatu sistem machine learning yang dapat mendeteksi email tersebut spam atau ham. Dengan adanya sistem tersebut diharapkan dapat membantu pengguna untuk lebih waspada terhadap email yang berbahaya |
| Metode pengolahan | Pengolahan yang digunakan pada proyek ini adalah tokenisasi fitur input (text email) dan juga mengubahnya menjadi huruf kecil. |
| Arsitektur model | Model yang digunakan pada proyek ini merupakan model LSTM sederhana dengan menggunakan Dense layer, Dropout layer, Pooling layer, Embedding layer, dan Input layer yang berisikan tokenisasi menggunakan Vectorization layer. |
| Metrik evaluasi | Metric yang digunakan pada model yaitu AUC, BinaryAccuracy, TruePositive, FalsePositive, TrueNegative, FalseNegative untuk mengevaluasi performa model sebuah klasifikasi |
| Performa model | Model yang dibuat menghasilkan performa yang cukup baik dalam memberikan prediksi, dan dari pelatihan yang dilakukan model menghasilkan binary_accuracy dan val_binary_accuracy di sekitar 95% |

