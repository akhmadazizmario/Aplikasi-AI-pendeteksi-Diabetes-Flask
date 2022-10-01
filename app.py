from flask import Flask, render_template, request, redirect  # render template
import pickle
import sklearn  # sklearn versi 0.0
import numpy as np  # numpy==1.19.3


# method flask name
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        # Data-data input
        model = pickle.load(open('knn_pickle', 'rb'))
        melahirkan = float(request.form['melahirkan'])
        glukosa = float(request.form['glukosa'])
        darah = float(request.form['darah'])
        kulit = float(request.form['kulit'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        riwayat = float(request.form['riwayat'])
        umur = float(request.form['umur'])

        # input hasil data
        datas = np.array((melahirkan, glukosa, darah, kulit,
                          insulin, bmi, riwayat, umur))
        datas = np.reshape(datas, (1, -1))

        # prediksi
        isDiabetes = model.predict(datas)

        # return ke view html
        return render_template('hasil.html', finalData=isDiabetes)
    else:
        # jika tidak berhasil dialihkan ke index
        return render_template('index.html')


# tampilan halaman utama
@app.route('/index')
def mastersuplier():
    return render_template('index.html')


# menjalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)
