# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:04:21 2020

手書き文字認識
手書き入力
"""

from PIL import Image, ImageDraw
from Svc import svc
from Neural_network import neural_network
import tkinter as tk
import glob
import pred_svc
import pred_neural_network

#保存時のデータ名変数
file_name = ""
#データの集計変数
cnt = 0
#ペンのサイズ
r = 10
#データ格納先
folder_path = "./data/"
#データの読み込み
files = glob.glob(folder_path + "*.png")
m = [glob.glob("./Svc/*.pkl"), glob.glob("./Neural_network/*.pkl")]


#データの集計
for file in files :
    cnt += 1


#キャンバスの作成
image = Image.new("RGB", (600, 600), (255, 255, 255))
draw = ImageDraw.Draw(image)



#手書き文字の読み込み変換及び評価学習
def image_convert() :
    global file_name, cnt
    
    #評価学習が終了するまで赤で表示
    label["bg"] = "red"
    
    #手書きで入力した文字の正解ラベルの入力
    ipt = tk.simpledialog.askstring("手書き入力", "手書き入力した文字を入力してください。")
    
    #未入力以外
    if ipt != None :
        result = {}
        tmp = []
        
        #手書き文字の保存 data_正解ラベル_データ総数.png
        file_name = folder_path + "data_" + ipt + "_" + str(cnt) + ".png"
        image.save(file_name)
        
        #ニューラルネットワークとSVCモデルの評価結果の保存
        tmp.append(pred_svc.predict_num(file_name))
        tmp.append(pred_neural_network.predict_num(file_name))
        
        
        #複数の評価結果の多数決
        for s in tmp :
            if s in result :
                result[s] += 1
            else :
                result[s] = 1
        
        
        max_key = max(result, key = result.get)
        
        #評価結果の表示
        label["text"] = str(max_key)
        #評価が終了したのでスカイブルーで表示
        label["bg"] = "skyblue"
        cnt += 1
        
        #評価データを含めた再学習        
        svc.svc_fit()
        neural_network.neural_network_fit()
        


#キャンバスの初期化
def clear() :
    global draw, image
    
    canvas.delete("oval")
    image = Image.new("RGB", (600, 600), (255, 255, 255))
    draw = ImageDraw.Draw(image)



idx = 0
mx = 0
my = 0
#マウスクリックの座標取得
def pressed(e) :
    global idx, mx, my
    idx = 1
    mx = e.x
    my = e.y



#マウス移動時の座標取得
def motion(e) :
    global mx, my
    mx = e.x
    my = e.y



#マウスクリックリリースイベント取得
def release(e) :
    global idx
    idx = 0



#メイン処理
def main() :
    global label
    
    #マウスがクリックされている間マウスポインタ―上に円を描画
    if idx == 1 :
        canvas.create_oval(mx - r, my - r, mx + r, my + r, fill = "black", tag = "oval")
        draw.ellipse((mx - r, my - r, mx + r, my + r), fill=(0, 0, 0), outline=(0, 0, 0))
    
    #1msの間隔で更新
    root.after(1, main)



#フレームの作成と描画
root = tk.Tk()

root.title("手書き入力")
root.geometry("1000x700")

canvas = tk.Canvas(root, width = 600, height = 600)
canvas["bg"] = "white"
canvas.bind("<Button>", pressed)
canvas.bind("<ButtonRelease>", release)
canvas.bind("<Motion>", motion)
canvas.place(x = 0, y = 50)

button1 = tk.Button(root, text = "クリア", width = 20, height = 1, command = clear)
button1.place(x = 10, y = 10)

button2 = tk.Button(root, text = "認識", width = 20, height = 1, command = image_convert)
button2.place(x = 210, y = 10)

label = tk.Label(root, text = "", width = 10, height = 1, font = ("Times New Roman", 20))
label["bg"] = "red"
label.place(x = 410, y = 10)


labels = []
cnt = 0
for i in m :
    for f in i :
        labels.append(tk.Label(root, text = f.split("\\")[1] + " : ", width = 50, height = 1))
        labels[cnt].place(x = 610, y = (50 + 30 * cnt))
        labels[cnt]["bg"] = "cyan"
        cnt += 1


main()
root.mainloop()










