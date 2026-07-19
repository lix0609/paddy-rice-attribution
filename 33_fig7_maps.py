# -*- coding: utf-8 -*-
"""Fig 7 (split): 4 张单产变化图 = {SSP2-4.5, SSP5-8.5} x {2041-2070, 2071-2100}。"""
import numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
 'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.titlesize':7.8})
MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PUB=PROJ+'/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-6,4),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
g=gpd.read_file(ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp').to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
mid=pd.read_csv(MID+'/projection_county_v2.csv'); mid['CountyCode']=mid['CountyCode'].astype(str)
e245=pd.read_csv(MID+'/projection_county_2071_245.csv'); e245['CountyCode']=e245['CountyCode'].astype(str)
e585=pd.read_csv(MID+'/projection_county_2071_585.csv'); e585['CountyCode']=e585['CountyCode'].astype(str)
panels=[('dY245',mid,'SSP2-4.5, 2041–2070','a'),('dY585',mid,'SSP5-8.5, 2041–2070','b'),
        ('dY',e245,'SSP2-4.5, 2071–2100','c'),('dY',e585,'SSP5-8.5, 2071–2100','d')]
vmax=max(abs(mid.dY245).max(),abs(mid.dY585).max(),abs(e245.dY).max(),abs(e585.dY).max())
fig,ax=plt.subplots(2,2,figsize=(DC*0.72,DC*0.68),constrained_layout=True)
axf=ax.ravel(); im=None
for a,(col,dfp,ttl,L) in zip(axf,panels):
    m=g.merge(dfp[['CountyCode',col]],on='CountyCode',how='left')
    im=m.plot(column=col,ax=a,cmap='RdBu',vmin=-vmax,vmax=vmax,edgecolor='0.5',lw=0.25).collections[0] if False else None
    m.plot(column=col,ax=a,cmap='RdBu',vmin=-vmax,vmax=vmax,edgecolor='0.5',lw=0.25)
    a.set_title(ttl); a.axis('off'); plab(a,L)
sm=mpl.cm.ScalarMappable(cmap='RdBu',norm=mpl.colors.Normalize(vmin=-vmax,vmax=vmax)); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,shrink=0.6,pad=0.02,location='right'); cb.set_label('Climate-driven Δyield (%)',fontsize=7)
fig.savefig(PUB+'/Fig7_proj_maps.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig7_proj_maps.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/Fig7_proj_maps.svg',bbox_inches='tight'); plt.close()
print('Fig7 maps saved | vmax=%.1f'%vmax)
