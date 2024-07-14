import numpy as np
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt
import soundfile as sf


def ov(data, samplerate, Fs, overlap):
    """オーバーラップをかける関数"""

    # 全データ長Ts, フレーム周期Fc, オーバーラップ時のフレームずらし幅s_ol, 平均化回数
    Ts = len(data) / samplerate
    Fc = Fs / samplerate
    x_ol = Fs * (1 - (overlap / 100))
    N_ave = int((Ts - (Fc * (overlap / 100))) /
                (Fc * (1 - (overlap / 100))))

    array = []

    # データを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)
        array.append(data[ps:ps + Fs:1])
        final_time = (ps + Fs)/samplerate
    return array, N_ave, final_time


def hanning(data_array, Fs, N_ave):
    """ハニング窓をかける関数（振幅補正係数計算付き）"""
    
    han = signal.windows.hann(Fs)
    acf = 1 / (sum(han) / Fs)

    # オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        data_array[i] = data_array[i] * han

    return data_array, acf


def db(x, dBref):
    """dB(デシベル）演算する関数"""
    
    y = 20 * np.log10(x / dBref)
    
    return y


def aweightings(f):
    """聴感補正(A補正)する関数"""
    
    if f[0] == 0:
        f[0] = 1
    ra = (np.power(12194, 2) * np.power(f, 4))/\
         ((np.power(f, 2) + np.power(20.6, 2)) *
          np.sqrt((np.power(f, 2) + np.power(107.7, 2)) *
                  (np.power(f, 2) + np.power(737.9, 2))) *
          (np.power(f, 2) + np.power(12194, 2)))
    a = 20 * np.log10(ra) + 2.00
    
    return a


def fft_ave(data_array, samplerate, Fs, N_ave, acf, no_db_a):
    """平均化FFTする関数"""
    
    fft_array = []
    fft_axis = np.linspace(0, samplerate, Fs)
    a_scale = aweightings(fft_axis)

    # FFTをして配列にdBで追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
    for i in range(N_ave):
        # dB表示しない場合とする場合で分ける
        if no_db_a == True:
            fft_array.append(acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2)))
        else:
            fft_array.append(db
                            (acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2))
                            , 2e-5))
    # 型をndarrayに変換しA特性をかける(A特性はdB表示しない場合はかけない）
    if no_db_a == True:
        fft_array = np.array(fft_array)
    else:
        fft_array = np.array(fft_array) + a_scale
    
    # 全てのFFT波形の平均を計算
    fft_mean = np.mean(np.sqrt(fft_array ** 2), axis=0)

    return fft_array, fft_mean, fft_axis


def plot_waveform(fft_array, final_time, samplerate):
    """スペクトログラムをプロットする関数"""

    # プロットの設定
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax = plt.subplots(figsize=(8, 5))

    # データをプロットする。
    im = ax.imshow(fft_array,
                    vmin=0, vmax=np.max(fft_array),
                    extent=[0, final_time, 0, samplerate],
                    aspect='auto',
                    cmap='jet')

    # カラーバーを設定する。
    cbar = fig.colorbar(im)
    cbar.set_label('SP [Pa]')

    # 軸設定する。
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    # スケールの設定をする。
    #ax.set_xticks(np.arange(0, 50, 1))
    #ax.set_yticks(np.arange(0, 20000, 200))
    #ax.set_xlim(0, 5)
    ax.set_ylim(0, 3000)
    fig.tight_layout()

    # グラフを表示する。
    plt.show()
    plt.close()
    
    return

if __name__ == '__main__':
    """メイン文"""
        
    # フレームサイズFsとオーバーラップ率overlapでスペクトログラムの分解能を調整する。
    Fs = 4096
    overlap = 90

    # wavファイルの読み込み
    data, samplerate = sf.read('recorded.wav')
    time_length = len(data) / samplerate

    # オーバーラップ抽出された時間波形配列
    time_array, N_ave, final_time = ov(data, samplerate, Fs, overlap)

    # ハニング窓関数をかける
    time_array, acf = hanning(time_array, Fs, N_ave)

    # FFTをかける
    fft_array, fft_mean, fft_axis = fft_ave(time_array, samplerate, Fs, N_ave, acf, no_db_a=False)

    # スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
    fft_array = fft_array.T

    # スペクトログラムをプロット
    plot_waveform(fft_array, final_time, samplerate)