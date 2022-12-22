from fileinput import filename
import sys
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,session,send_from_directory
import tensorflow as tf
import os
from tensorflow import keras
import csv
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)
pre=keras.models.load_model("model1l.h5")
app.secret_key="heydude"
@app.route('/')
def h():
    return render_template('ui.html')
@app.route('/predict',methods=['POST','GET'])
def predict(methods=['POST','GET']):
    b=[]
    if request.method=='POST':
        pre=keras.models.load_model("model1l.h5")
        b.append([float(request.form['v1'])])
        b.append([float(request.form['v2'])])
        b.append([float(request.form['v3'])])
        b.append([float(request.form['v4'])])
        b.append([float(request.form['v5'])])
        b.append([float(request.form['v6'])])
        b.append([float(request.form['v7'])])
        b.append([float(request.form['v8'])])
        b.append([float(request.form['v9'])])
        b.append([float(request.form['v10'])])
        b.append([float(request.form['v11'])])
        b.append([float(request.form['v12'])])
        b.append([float(request.form['v13'])])
        b.append([float(request.form['v14'])])
        b.append([float(request.form['v15'])])
        b.append([float(request.form['v16'])])
        b.append([float(request.form['v17'])])
        b.append([float(request.form['v18'])])
        b.append([float(request.form['v19'])])
        b.append([float(request.form['v20'])])
        b.append([float(request.form['v21'])])
        b.append([float(request.form['v22'])])
        b.append([float(request.form['v23'])])
        b.append([float(request.form['v24'])])
        b.append([float(request.form['v25'])])
        b.append([float(request.form['v26'])])
        b.append([float(request.form['v27'])])
        b.append([float(request.form['v28'])])
        b.append([float(request.form['am'])])
        s=StandardScaler()
        b=s.fit_transform(b)
        bb=[]
        
        for j in b:
            bb.extend(j)
        p=pre.predict([bb])
        if p>=0.5:
            return render_template("cc.html",outt='It is a fraudelent transaction',outtt=' ')
        else:
            return render_template("cc.html",outt='It is a legitimate transaction',outtt=' ')
@app.route("/files",methods=['POST','GET'])
def files(methods=['POST','GET']):
    def ch(l:int):
        if len(s)!=29:
                return render_template("ccn.html",outt="Upload csv file with exactly 29 required inputs",outtt=' ')
    def check_datatype(x:list):
        for j in range(0,len(x),1):
            for ind,v in enumerate(x[j]):
                if isinstance(x[j][ind],str):
                    return 0
                    break
                pass
        return 1
    def scale(x:list):
        s=StandardScaler()
        x=s.fit_transform(x)
        return x
    def setsession(gg:str,dd):
        g="templates/"+gg
        dd.to_csv(g,index=False)
        session['fnn']=g
        session['ff']=gg
    if request.method=='POST':
        rr=request.files['file']
        rrr = str(rr.filename)
        if rr.filename=='' or rrr[len(rrr)-4::1]!='.csv':
            return render_template("ccn.html",outt='Upload csv file',outtt=' ')
        else:
            filename=secure_filename(rr.filename)
            n_filename=f'{filename.split(".")[0]}_fff.csv'
            save_loc=os.path.join('Filescsv',n_filename)
            rr.save(save_loc)
            df=pd.read_csv(str(save_loc))
            s=df.columns.values
            ch(len(s))   
            
            if (s[0][0:1]>='a' and s[0][0:1]<='z') or (s[0][0:1]>='A' and s[0][0:1]<='Z'):
                x=df.iloc[0:,0:].values
                if (check_datatype(x))==0:
                    return render_template("ccn.html",outt='Inputs should be numbers',outtt='')
                x=scale(x)
                if df.shape[0]==1:
                    predicted=pre.predict(x)
                    predicted='Fraud' if predicted>=0.5 else 'Legitimate'
                    df['predicted']=predicted
                    setsession(str('Predicted_'+n_filename),df)
                    if predicted==1:
                        return render_template("cc.html",outt="It is a fraudelent transaction")
                    return render_template("cc.html",outt="It is a legitimate transaction")
                elif df.shape[0]>=2:
                    p=np.array(pre.predict(x))
                    p=np.where(p>=0.5,'Fraud','Legitimate')
                    p=np.ravel(p)
                    dd=pd.read_csv(str(save_loc))
                    dd['predicted']=p
                    dd['predicted']=dd['predicted'].astype('category')
                    setsession(str('Predicted_'+n_filename),dd)
                    return redirect(url_for("table"))
                else:
                    return render_template("ccn.html",outt="Upload csv file with atleast 1 record",outtt=" ")    
            else:
                
                a=df.columns.values
                for ind,it in enumerate(a):
                    if float(it):
                        a[ind]=float(it)
                    else:
                        return render_template("ccn.html",outt='Inputs should be numbers',outtt='')
                df.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29']
                df.loc[len(df)]=a
                x=df.iloc[:,:].values
                if(check_datatype(x))==0:
                    return render_template("ccn.html",outt='Inputs should be numbers',outtt='')
                if df.shape[0]==1:
                    x=scale(x)
                    if any(x) is str or (any(x) is not int and any(x) is not float):
                        return render_template("ccn.html",outt="Inputs must be numbers or decimals",outtt='')
                    predicted=pre.predict(x)
                    predicted='Fraud' if predicted>=0.5 else 'Legitimate'
                    df['predicted']=predicted
                    setsession(str('Predicted_'+n_filename),df)
                    
                    if predicted==1:
                        return render_template("cc.html",outt="It is a fraudelent transaction")
                    else:
                        return render_template("cc.html",outt="It is a legitimate transaction")
                elif df.shape[0]>=2:
                    x=scale(x)
                    p=np.array(pre.predict(x))
                    p=np.where(p>=0.5,'Fraud','Legitimate')
                    p=np.ravel(p)
                    dd=pd.read_csv(str(save_loc))
                    a=dd.columns.values
                    for ind,it in enumerate(a):
                        if float(it):
                            a[ind]=float(it)
                    dd.columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29']
                    dd.loc[len(df)]=a
                    dd['predicted']=p
                    dd['predicted']=dd['predicted'].astype('category')
                    
                    setsession(str('Predicted_'+n_filename),dd)
                    return redirect(url_for("table"))
                else:
                    return render_template("ccn.html",outt='Upload csv file with atleast one record',outtt=' ')
@app.route("/table")
def table():
    if 'fnn' in session:
        nm=session['fnn']
        df1=pd.read_csv(nm)
        return render_template("table.html",tables=[df1.to_html()],titles=[''])
@app.route("/dw")
def dw():
    if 'ff' in session:
        da=session['ff']
        return send_from_directory('templates',da)

if __name__=='__main__':
    app.run(debug=True)