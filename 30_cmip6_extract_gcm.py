# -*- coding: utf-8 -*-
"""逐县提取72个GCM(9模式×tas/pr×2窗×2情景)月度数据 -> cmip6_gcm_monthly.csv。"""
import os,re,glob,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'
UNC=PROJ+'/02_中间数据/cmip6/不确定性'; OUT=PROJ+'/02_中间数据'
SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'
BASE=PROJ+'/02_中间数据/cmip6/cmip6_urls_ensemble/China_tas_1km_1991-2020/China_tas_1km_1991-2020.tif'
tifs=[]
for root,_,fs in os.walk(UNC):
    for fn in fs:
        if fn.endswith('.tif'): tifs.append(os.path.join(root,fn))
print('GCM tif数',len(tifs))
def key(fn):
    m=re.search(r'China_([a-z]+)_(.+?)_r1i1p1f1_1km_(\d{4}-\d{4})_(SSP\d+)',os.path.basename(fn))
    v,mo,w,sp=m.groups(); win={'2041-2070':'2041','2071-2100':'2071'}[w]; sspc={'SSP245':'245','SSP585':'585'}[sp]
    return '%s_%s_%s_%s'%(v,mo,win,sspc)
g=gpd.read_file(SHP).to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
BB=(121.0,40.7,131.7,46.6)
with rasterio.open(BASE) as s:
    win=from_bounds(*BB,s.transform).round_offsets().round_lengths()
    wt=s.window_transform(win); H=int(win.height); W=int(win.width); nod=s.nodata
labels=rasterize([(geom,i+1) for i,geom in enumerate(g.geometry)],out_shape=(H,W),transform=wt,fill=0,dtype='int32')
rows=[]
for k,path in sorted((key(t),t) for t in tifs):
    with rasterio.open(path) as s:
        win=from_bounds(*BB,s.transform).round_offsets().round_lengths()
        arr=s.read(window=win).astype('float32'); arr[arr==nod]=np.nan; arr[arr<-1e30]=np.nan
    for i,code in enumerate(g['CountyCode']):
        m=labels==(i+1)
        if m.sum()==0: continue
        for mon in range(min(12,arr.shape[0])):
            v=arr[mon][m]; v=v[np.isfinite(v)]
            rows.append((code,k,mon+1,np.nan if v.size==0 else float(v.mean())))
df=pd.DataFrame(rows,columns=['CountyCode','dataset','month','value'])
piv=df.pivot_table(index=['CountyCode','dataset'],columns='month',values='value').reset_index()
piv.columns=['CountyCode','dataset']+['m%02d'%m for m in range(1,13)]
piv.to_csv(OUT+'/cmip6_gcm_monthly.csv',index=False,encoding='utf-8-sig')
print('saved cmip6_gcm_monthly.csv datasets=%d 县=%d'%(piv.dataset.nunique(),piv.CountyCode.nunique()))
print('样例7月省均:')
for k in ['tas_ACCESS-CM2_2041_585','tas_MIROC6_2071_585','tas_GFDL-ESM4_2071_585']:
    print(' %-28s %.2f'%(k,piv[piv.dataset==k]['m07'].mean()))
