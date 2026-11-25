"""Inspect the installed Qt controls, scope dispatch, old-project counts and exported workbook."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import tifffile
from openpyxl import load_workbook
from napari.qt import get_qapp
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QApplication, QPushButton, QGroupBox
from cfos.core.project import Project, ImageEntry
from cfos.core.sections import Section
from cfos.core.export import export_excel
from cfos.core.pipeline import refresh_coloc
from cfos.core.detection import ChannelParams
from cfos.gui.main_window import MainWindow

def check_gui(folder):
    app=get_qapp()
    image=folder/'sample.ome.tif'
    tifffile.imwrite(image,np.ones((2,64,128),np.uint16),ome=True,metadata={'axes':'CYX','PhysicalSizeX':1.,'PhysicalSizeY':1.})
    im=ImageEntry(str(image),[Section(i,(i*64,0,64,64),[],f'S0{i+1}') for i in range(2)])
    p=Project(folder,images=[im],channel_names=['DAPI','Green'],grading={'Green':[2.,4.]})
    p.stats={'metric':'Green+ 强 密度 (个/mm²)'}
    p.detection.channels[1]=ChannelParams(min_intensity=9999)
    p.save()
    df=pd.DataFrame([dict(section=sec.name,channel=c,x=i*64.+12,y=14.,contour=[],cell_id=2*i+c,merged_id=i,combo='0+1',area_um2=25.,mean_ch0=100.,mean_ch1=110.,background=15.,grade_ch1=3) for i,sec in enumerate(im.sections) for c in (0,1)])
    df.attrs['processed_sections']=['S01','S02'];p.save_cells(im,df)
    w=MainWindow();w._load_project(Project.load(folder))
    w.steps.setCurrentRow(3)
    buttons={b.text():b for b in w.findChildren(QPushButton)}
    assert all(name not in buttons for name in ('识别已选图片和脑片','按新阈值重新筛选（几秒钟，不用重新识别）','按全部细胞三等分建议','应用并画分布图'))
    assert all('阳性细胞强度分级' not in g.title() for g in w.findChildren(QGroupBox))
    go=buttons['选择处理范围并识别…']
    def choose():
        dlg=QApplication.activeModalWidget()
        dlg.tree.topLevelItem(0).child(1).setCheckState(0,Qt.Checked);dlg.ok.click()
    with patch.object(w,'run_detection') as run:
        QTimer.singleShot(20,lambda:QApplication.activeModalWidget().reject());go.click();run.assert_not_called()
        QTimer.singleShot(20,choose);go.click();run.assert_called_once_with(True)
    assert w._processing_scope=={im.key:('S02',)}
    w.steps.setCurrentRow(4)
    assert '重新计算所选脑片的共标' in buttons
    assert refresh_coloc(w.project,selection=w._processing_scope)==1
    assert len(w.project.load_cells(im))==4
    w._refresh_coloc_table()
    cols=[w.coloc_table.horizontalHeaderItem(i).text() for i in range(w.coloc_table.columnCount())]
    assert 'Green+' in cols and 'DAPI+Green 共标' in cols and 'Green+ 强' not in cols
    w.steps.setCurrentRow(6)
    assert '强' not in w.st_metric.currentText()
    dest=folder/'out.xlsx'
    export_excel(w.project,dest,masks=False,qc=False,quality=False)
    wb=load_workbook(dest,read_only=True)
    try:
        sheet=wb['按脑片汇总']; rows=list(sheet.values); names=list(rows[0])
        assert 'Green+' in names and all(r[names.index('Green+')]==1 for r in rows[1:])
        assert all('Green+ 强' not in str(v) and '强度分级' not in str(v) for ws in wb for row in ws.values for v in row)
    finally:wb.close()
    w.close();app.processEvents()
    print('PASS scope accept/cancel, removed Qt actions, existing positivity and colocalization, groups, XLSX',flush=True)
