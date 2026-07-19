# -*- coding: utf-8 -*-
"""可续跑：逐年提取分县水田面积，已完成年份跳过。"""
import os, glob, re, sys, time, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
warnings.filterwarnings('ignore')
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'
PADDY=ROOT+'/水稻数据集/Long history paddy rice mapping across Northeast China with deep learning and annual result enhancement method/submission'
SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'
OUT=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'
CSV=OUT+'/paddy_area_by_county_1985_2020.csv'
PIX_HA=0.09
BUDGET=float(sys.argv[1]) if len(sys.argv)>1 else 38.0  # 秒预算

gdf=gpd.read_file(SHP); gdf['CountyCode']=gdf['CountyCode'].astype(str)
tifs={}
for f in glob.glob(PADDY+'/*.tif'):
    m=re.match(r'^(\d{4})$',os.path.splitext(os.path.basename(f))[0])
    if m and 1985<=int(m.group(1))<=2020: tifs[int(m.group(1))]=f
done=set()
if os.path.exists(CSV):
    try: done=set(pd.read_csv(CSV)['year'].unique().tolist())
    except: pass
todo=[y for y in sorted(tifs) if y not in done]
print('done:',sorted(done)); print('todo:',todo)
t0=time.time(); newrows=[]
for y in todo:
    if time.time()-t0>BUDGET: print('budget reached, stop before',y); break
    with rasterio.open(tifs[y]) as src:
        gg=gdf.to_crs(src.crs)
        for code,name,geom in zip(gg['CountyCode'],gg['Name'],gg.geometry):
            try:
                arr,_=mask(src,[geom],crop=True,filled=True,nodata=3)
                px=int((arr[0]==1).sum())
            except: px=0
            newrows.append((y,code,name,px,px*PIX_HA))
    print('extracted',y,'%.1fs'%(time.time()-t0))
if newrows:
    dfn=pd.DataFrame(newrows,columns=['year','CountyCode','Name','paddy_pix','paddy_ha'])
    if os.path.exists(CSV):
        dfn=pd.concat([pd.read_csv(CSV),dfn],ignore_index=True)
    dfn.sort_values(['year','CountyCode']).to_csv(CSV,index=False,encoding='utf-8-sig')
    print('saved rows:',len(dfn))
print('remaining:',[y for y in sorted(tifs) if y not in set(pd.read_csv(CSV)['year'].unique())] if os.path.exists(CSV) else tifs)
