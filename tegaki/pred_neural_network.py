# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 07:23:40 2020

手書き文字認識　オリジナル
ニューラルネットワークバージョン
評価
"""

from PIL import Image
import joblib
from matplotlib import pylab as plt
import numpy as np
import glob
import common_variable as cv

#データ格納先
folder_path = "./data/"
#データの読み込み
files = glob.glob(folder_path + "*.png")
flg = "n"
#学習サイズの読み込み
size = cv.img_size()


#学習モデルの読み込み
model = []
files2 = glob.glob("./Neural_network/*.pkl")
for file in files2 :
    model.append(joblib.load(file))



#学習画像の表示
def plt_show(png_file) :
    #画像読み込み
    img = Image.open(png_file)
    #リサイズ
    img.thumbnail(size, Image.LANCZOS)
    #グレイスケール
    img = img.convert("L")
    #表示
    plt.imshow(img, cmap = "gray")




#画像ファイルを指定して判定する（24bit PNG）
def predict_num(png_file) :
    #評価結果格納用
    res = {}
    r = []
    cnt = 0
    
    #画像読み込み
    img = Image.open(png_file)
    #リサイズ
    img.thumbnail(size, Image.LANCZOS)
    #グレイスケール
    img = img.convert("L")
        
    #画像→配列
    img_np = np.array(img, "f")
    #ネガポジ反転
    img_np = 255 - img_np
    #0～16の範囲に揃える
    img_np = img_np // 16
    #一次元に変換
    img_np = img_np.reshape(-1,)
    
    
    #複数モデルによる多数決
    for m in model :
        #評価結果の格納
        r.append(m.predict([img_np]))
        
        if r[cnt][0] in res :
            res[r[cnt][0]] += 1
        else :
            res[r[cnt][0]] = 1
    
    
    max_key = max(res, key = res.get)
    
    return max_key



#メイン処理
def main() :
    true_cnt = 0
    cnt = 0
    
    
    #テスト画像で判定してみる
    for file in files :
        cnt += 1
        result = predict_num(file)
            
        if result == int(file.split("_")[1].replace(".png", "")) :
            true_cnt += 1



#直接実行
if __name__ == "__main__" :
    #y:ファイル入力をして評価 n:テストデータの評価
    ipt = input("入力　[y/n] : ")
    
    if ipt == "y" :
        flg = "y"
        path = folder_path + input("読み込みファイル名 : ")
        print(predict_num(path))
    elif ipt == "n" :
        main()
    else :
        print("入力に誤りがあります")

















