# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:48:15 2020

ニューラルネットワーク
手書き文字認識
学習
"""

from PIL import Image
import joblib
from sklearn.neural_network import MLPClassifier
from .. import common_variable as cv
import numpy as np
import glob

#学習データの格納先
folder_path = "./data/"
#モデルの格納先
model_path = "./Neural_network/model_neural_network"
#モデルの数
n = 3
#学習サイズ
size = cv.img_size()



#学習
def neural_network_fit() :
    #データの読み込み
    files = glob.glob(folder_path + "*.png")
    
    
    
    #画像の変換
    def file_convert(png_file) :
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
        
        return img_np
    
    
    
    x = []
    y = []
    
    
    #学習データと正解ラベルの設定
    for file in files :        
        x.append(file_convert(file))        
        y.append(file.split("_")[1])

    
    #n回モデルの生成
    for i in range(n) :
        #モデル作成
        model = MLPClassifier(hidden_layer_sizes = (16, ))
        #学習
        model.fit(x, y)
        #学習済みモデルの保存                
        joblib.dump(model, model_path + str(i) + ".pkl", compress = True)



#直接実行した場合
if __name__ == "__main__" :
    model_path = "./model_neural_network_"
    folder_path = "../data/"
    neural_network_fit()




