# -*- coding: utf-8 -*-
"""逐县提取CMIP6月度tas/pr: 基准1991-2020 + 三未来窗(2021-2040/2041-2070/2071-2100) x SSP245/585 (ESM)。"""
import os,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'
CMD=PROJ+'/02_中间数据/cmip6'; OLD=CMD+'/cmip6_urls_ensemble'; NEW=CMD+'/ESM 集合均值'; OUT=PROJ+'/02_中间数据'
SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'
def esm(win,ssp,var,root):
    n='China_%s_ESM_r1i1p1f1_1km_%s_%s'%(var,win,ssp); return os.path.join(root,n,n+'.tif')
FILES={'tas_base':os.path.join(OLD,'China_tas_1km_1991-2020/China_tas_1km_1991-2020.tif'),
       'pr_base':os.path.join(OLD,'China_pr_1km_1991-2020/China_pr_1km_1991-2020.tif')}
for var in ['tas','pr']:
    for ssp,tag in [('SSP245','245'),('SSP585','585')]:
        FILES['%s_%s_2021'%(var,tag)]=esm('2021-2040',ssp,var,NEW)
        FILES['%s_%s_2041'%(var,tag)]=esm('2041-2070',ssp,var,OLD)
        FILES['%s_%s_2071'%(var,tag)]=esm('2071-2100',ssp,var,NEW)
missing=[k for k,v in FILES.items() if not os.path.exists(v)]
print('缺失文件:',missing if missing else '无')
g=gpd.read_file(SHP).to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
BB=(121.0,40.7,131.7,46.6)
with rasterio.open(FILES['tas_base']) as s:
    win=from_bounds(*BB,s.transform).round_offsets().round_lengths()
    wt=s.window_transform(win); H=int(win.height); W=int(win.width); nod=s.nodata
labels=rasterize([(geom,i+1) for i,geom in enumerate(g.geometry)],out_shape=(H,W),transform=wt,fill=0,dtype='int32')
print('吉林窗口 %dx%d 覆盖像元'%(H,W),np.count_nonzero(labels))
rows=[]
for key,path in FILES.items():
    with rasterio.open(path) as s:
        win=from_bounds(*BB,s.transform).round_offsets().round_lengths()
        arr=s.read(window=win).astype('float32'); arr[arr==nod]=np.nan; arr[arr<-1e30]=np.nan
    for i,code in enumerate(g['CountyCode']):
        m=labels==(i+1)
        if m.sum()==0: continue
        for mon in range(min(12,arr.shape[0])):
            v=arr[mon][m]; v=v[np.isfinite(v)]
            rows.append((code,key,mon+1,np.nan if v.size==0 else float(v.mean())))
df=pd.DataFrame(rows,columns=['CountyCode','dataset','month','value'])
piv=df.pivot_table(index=['CountyCode','dataset'],columns='month',values='value').reset_index()
piv.columns=['CountyCode','dataset']+['m%02d'%m for m in range(1,13)]
piv.to_csv(OUT+'/cmip6_county_monthly_allwin.csv',index=False,encoding='utf-8-sig')
print('saved cmip6_county_monthly_allwin.csv datasets=%d 县=%d'%(piv.dataset.nunique(),piv.CountyCode.nunique()))
for key in ['tas_base','tas_245_2021','tas_245_2041','tas_245_2071','tas_585_2071']:
    print('%-14s 7月省均=%.2f'%(key,piv[piv.dataset==key]['m07'].mean()))
