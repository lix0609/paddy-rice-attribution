# -*- coding: utf-8 -*-
"""逐县提取CMIP6月度tas/pr (基准1991-2020 + 未来2041-2070 SSP245/585)。"""
import os,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'
CM=PROJ+'/02_中间数据/cmip6/cmip6_urls_ensemble'; OUT=PROJ+'/02_中间数据'
SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'
FILES={
 'tas_base':'China_tas_1km_1991-2020/China_tas_1km_1991-2020.tif',
 'pr_base':'China_pr_1km_1991-2020/China_pr_1km_1991-2020.tif',
 'tas_245':'China_tas_ESM_r1i1p1f1_1km_2041-2070_SSP245/China_tas_ESM_r1i1p1f1_1km_2041-2070_SSP245.tif',
 'tas_585':'China_tas_ESM_r1i1p1f1_1km_2041-2070_SSP585/China_tas_ESM_r1i1p1f1_1km_2041-2070_SSP585.tif',
 'pr_245':'China_pr_ESM_r1i1p1f1_1km_2041-2070_SSP245/China_pr_ESM_r1i1p1f1_1km_2041-2070_SSP245.tif',
 'pr_585':'China_pr_ESM_r1i1p1f1_1km_2041-2070_SSP585/China_pr_ESM_r1i1p1f1_1km_2041-2070_SSP585.tif',
}
g=gpd.read_file(SHP).to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
BB=(121.0,40.7,131.7,46.6)  # 吉林bbox
# 用第一个tif定义吉林窗口网格
p0=os.path.join(CM,FILES['tas_base'])
with rasterio.open(p0) as s:
    win=from_bounds(*BB,s.transform); win=win.round_offsets().round_lengths()
    wt=s.window_transform(win); H=int(win.height); W=int(win.width)
    nod=s.nodata
# 栅格化县界到该网格(label=行号+1)
shapes=[(geom,i+1) for i,geom in enumerate(g.geometry)]
labels=rasterize(shapes,out_shape=(H,W),transform=wt,fill=0,dtype='int32')
print('吉林窗口 %dx%d, 覆盖县像元数:'%(H,W), np.count_nonzero(labels))
rows=[]
for key,rel in FILES.items():
    with rasterio.open(os.path.join(CM,rel)) as s:
        win=from_bounds(*BB,s.transform).round_offsets().round_lengths()
        arr=s.read(window=win).astype('float32')  # (12,H,W)
        arr[arr==nod]=np.nan; arr[arr<-1e30]=np.nan
    for i,code in enumerate(g['CountyCode']):
        m=labels==(i+1)
        if m.sum()==0: continue
        for mon in range(12):
            v=arr[mon][m]; v=v[np.isfinite(v)]
            rows.append((code,key,mon+1,np.nan if v.size==0 else float(v.mean())))
df=pd.DataFrame(rows,columns=['CountyCode','dataset','month','value'])
piv=df.pivot_table(index=['CountyCode','dataset'],columns='month',values='value').reset_index()
piv.columns=['CountyCode','dataset']+['m%02d'%m for m in range(1,13)]
piv.to_csv(OUT+'/cmip6_county_monthly.csv',index=False,encoding='utf-8-sig')
print('saved cmip6_county_monthly.csv rows=%d 县=%d'%(len(piv),piv.CountyCode.nunique()))
# 快检:各数据集7月均温/降水(省均)
for key in FILES:
    sub=piv[piv.dataset==key]['m07']
    print('%-9s 7月 省均=%.1f'%(key,sub.mean()))
