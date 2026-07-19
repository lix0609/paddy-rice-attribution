# -*- coding: utf-8 -*-
"""逐年水田分县面积提取 + 县静态属性(质心纬度/经度、平均高程)。"""
import os, glob, re, warnings
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from rasterio.mask import mask
warnings.filterwarnings('ignore')

ROOT = '/sessions/vibrant-bold-franklin/mnt/水稻'
PADDY_DIR = os.path.join(ROOT, '水稻数据集/Long history paddy rice mapping across Northeast China with deep learning and annual result enhancement method/submission')
SHP = os.path.join(ROOT, '吉林省_区县行政边界/吉林省_区县_合并50.shp')
DEM = os.path.join(ROOT, '3.空间数据预处理/DEM/dem_jl.tif')
OUT = os.path.join(ROOT, '增温驱动吉林水田扩张与增产_1985-2020/02_中间数据')
os.makedirs(OUT, exist_ok=True)

PIX_HA = 30*30/10000.0  # 0.09 ha/像元

# 县界
gdf = gpd.read_file(SHP)
gdf['CountyCode'] = gdf['CountyCode'].astype(str)

# 静态属性: 质心经纬度(4326)
g4326 = gdf.to_crs(4326)
cent = g4326.geometry.centroid
static = pd.DataFrame({'CountyCode': gdf['CountyCode'].values,
                       'Name': gdf['Name'].values,
                       'City': gdf['City'].values,
                       'lon': cent.x.values, 'lat': cent.y.values})

# 县平均高程
with rasterio.open(DEM) as dem:
    gdem = gdf.to_crs(dem.crs)
    elevs = []
    for geom in gdem.geometry:
        try:
            arr, _ = mask(dem, [geom], crop=True, filled=True, nodata=np.nan)
            a = arr[0].astype('float32')
            nod = dem.nodata
            if nod is not None: a = np.where(a==nod, np.nan, a)
            elevs.append(np.nanmean(a))
        except Exception:
            elevs.append(np.nan)
static['elev_mean'] = elevs
static.to_csv(os.path.join(OUT, 'county_static.csv'), index=False, encoding='utf-8-sig')
print('county_static saved; elev range', np.nanmin(elevs), np.nanmax(elevs))

# 逐年水田面积
tifs = {}
for f in glob.glob(os.path.join(PADDY_DIR, '*.tif')):
    m = re.match(r'^(\d{4})$', os.path.splitext(os.path.basename(f))[0])
    if m:
        y = int(m.group(1))
        if 1985 <= y <= 2020:
            tifs[y] = f
years = sorted(tifs)
print('years:', years)

rows = []
for y in years:
    with rasterio.open(tifs[y]) as src:
        gg = gdf.to_crs(src.crs)
        for code, name, geom in zip(gg['CountyCode'], gg['Name'], gg.geometry):
            try:
                arr, _ = mask(src, [geom], crop=True, filled=True, nodata=3)
                a = arr[0]
                paddy = int(np.sum(a == 1))
            except Exception:
                paddy = 0
            rows.append((y, code, name, paddy, paddy*PIX_HA))
    print('done', y)

df = pd.DataFrame(rows, columns=['year','CountyCode','Name','paddy_pix','paddy_ha'])
df.to_csv(os.path.join(OUT, 'paddy_area_by_county_1985_2020.csv'), index=False, encoding='utf-8-sig')

# 省级汇总
prov = df.groupby('year')['paddy_ha'].sum().reset_index()
prov.to_csv(os.path.join(OUT, 'paddy_area_province.csv'), index=False, encoding='utf-8-sig')
print(prov.to_string(index=False))
print('ALL DONE')
