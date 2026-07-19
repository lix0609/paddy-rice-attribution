import pandas as pd,numpy as np,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':0.7})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00'}; MM=1/25.4
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PUB=PROJ+'/03_figures/pub'
g=gpd.read_file(ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp').to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
cc=pd.read_csv(MID+'/county_change_summary.csv'); cc['CountyCode']=cc['CountyCode'].astype(str)
pcj=pd.read_csv(MID+'/projection_county.csv'); pcj['CountyCode']=pcj['CountyCode'].astype(str)
fig=plt.figure(figsize=(133*MM,88*MM))
gs=fig.add_gridspec(2,2,height_ratios=[0.34,1],hspace=0.34,wspace=0.34)
# ---- top: method pipeline (full width) ----
axp=fig.add_subplot(gs[0,:]); axp.axis('off'); axp.set_xlim(0,1); axp.set_ylim(0,1)
boxes=[('Satellite paddy map\n+ statistics + climate',0.16,'#eef3f8',OI['blue']),
       ('Interpretable ML (RF+SHAP)\n+ LMDI + attribution',0.5,'#eef8f2',OI['green']),
       ('CMIP6 delta-change\nprojection (2041–2070)',0.84,'#fdf1e8',OI['orange'])]
for t,x,fc,ec in boxes:
    axp.text(x,0.5,t,ha='center',va='center',fontsize=7,linespacing=1.3,
             bbox=dict(boxstyle='round,pad=0.35',fc=fc,ec=ec,lw=0.8))
for x0,x1 in [(0.29,0.35),(0.63,0.69)]:
    axp.annotate('',xy=(x1,0.5),xytext=(x0,0.5),arrowprops=dict(arrowstyle='-|>',color='0.4',lw=1.3))
# ---- bottom-left: historical expansion ----
a0=fig.add_subplot(gs[1,0]); m=g.merge(cc[['CountyCode','expansion_x']],on='CountyCode',how='left'); m['e']=m['expansion_x'].clip(upper=8)
m.plot(column='e',ax=a0,cmap='YlOrRd',edgecolor='0.6',lw=0.2); a0.axis('off')
a0.set_title('Past: 1985–2020',fontsize=8,fontweight='bold',pad=1)
a0.text(0.5,-0.06,'Climate-enabled expansion\nPaddy 3.1×, production 3.3× (area 69%)\n+0.74°N, −99 m; yield tech-led',
        transform=a0.transAxes,ha='center',va='top',fontsize=6.3,linespacing=1.35)
# ---- bottom-right: future projection ----
a2=fig.add_subplot(gs[1,1]); mp=g.merge(pcj[['CountyCode','dY_585']],on='CountyCode',how='left'); vmax=abs(pcj['dY_585']).max()
mp.plot(column='dY_585',ax=a2,cmap='RdBu',vmin=-vmax,vmax=vmax,edgecolor='0.6',lw=0.2); a2.axis('off')
a2.set_title('Future: 2041–2070',fontsize=8,fontweight='bold',pad=1)
a2.text(0.5,-0.06,'Expansion↑ but yield↓\nSuitability 80% → 98–100%↑\nClimate-driven yield −1.5 to −3.0%↓',
        transform=a2.transAxes,ha='center',va='top',fontsize=6.3,linespacing=1.35)
fig.suptitle('Climate drove the expansion, technology drove the intensification — warming now splits them',fontsize=8,y=1.0)
fig.savefig(PUB+'/Graphical_Abstract.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Graphical_Abstract.png',dpi=600,bbox_inches='tight'); plt.close()
print('Graphical Abstract redone')
