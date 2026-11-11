"""v11.48: пакет-дополнение с книжными коврами. Без браузера: python3 tests/test_books_bundle.py"""
import os, sys, json, time, zipfile, io, shutil, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); CPS = os.path.dirname(HERE)
tmp = tempfile.mkdtemp(prefix='cps_books_'); app = os.path.join(tmp, 'cps')
shutil.copytree(CPS, app, ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc', 'rug_lib', 'rug_img', 'rugs', 'rug_thumbs'))
os.environ.update(CPS_RUGS_TEST='1', CPS_PASSWORD='', CPS_NO_SCHED='1', CPS_MOTIF_FAKE='1', CPS_SAM='0', CPS_VLM_MOCK='1', CPS_GITHUB_TOKEN='')
sys.path.insert(0, app); os.chdir(app)
from PIL import Image
from fastapi.testclient import TestClient
import backend
c = TestClient(backend.app)
img = io.BytesIO(); Image.new('RGB', (900, 1400), (150, 40, 40)).save(img, 'JPEG'); img = img.getvalue()

def bundle(path, cat):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('catalog.json', json.dumps(cat, ensure_ascii=False))
        for it in cat['items']: z.writestr(it['img'], img)

def sync(path):
    r = c.post('/api/rugs_sync', json={'bundle_id': path}); assert r.status_code == 200, r.text
    for _ in range(240):
        st = c.get('/api/rugs_sync').json()
        if not st.get('running'): assert not st.get('error'), st; return st
        time.sleep(0.25)
    raise AssertionError('синхронизация не закончилась')

MUS = 'Национальный музей азербайджанского ковра'
bundle(f'{tmp}/main.zip', {'built': 'main1', 'groups': [], 'items': [
    {'id': 'azcm_143', 'src': 'azcarpetmuseum', 'title': 'Palaz', 'museum': MUS, 'inv': '1614', 'img': 'img/azcm_143.jpg'}]})
sync(f'{tmp}/main.zip'); main_bid = c.get('/api/rugs_sync').json()['bundle_id']
B = 'azerbaycan_xalchasi_ensiklopediyasi_ii_cild'
bundle(f'{tmp}/books.zip', {'built': 'books1', 'extra': True, 'groups': [], 'items': [
    {'id': f'{B}_p0032_1', 'src': 'book:' + B, 'img': f'img/{B}_p0032_1.jpg', 'title': 'Çiyi palaz Ширван', 'museum': MUS, 'inv': '1614',
     'size': '142x220 см', 'school': 'guba-shirvan', 'book': B, 'page': 32, 'text': 'Çiyi Palaz. Şirvan. … DDK 1614.'},
    {'id': 'azerbaycan_xalchalari_qarabag_qrupu_p0125_1', 'src': 'book:azerbaycan_xalchalari_qarabag_qrupu', 'img': 'img/m_p0125_1.jpg',
     'title': 'Xalça. Qarabağ qrupu', 'object': 'современная копия (Азер-ильме)', 'book': 'azerbaycan_xalchalari_qarabag_qrupu', 'page': 125}]})
sync(f'{tmp}/books.zip')
st = c.get('/api/rugs_sync').json()
assert st['bundle_id'] == main_bid, f'пакет-дополнение затёр основной id: {st["bundle_id"]}'
r = c.get(f'/api/rugs/{B}_p0032_1').json()
assert r['cat'].get('book') == B and r['cat'].get('page') == 32, r['cat']
assert any(x['id'] == 'azcm_143' and x.get('inv') for x in r.get('same') or []), f'нет двойника по номеру DDK: {r.get("same")}'
m = c.get('/api/rugs/azcm_143').json()
assert any(x['id'] == f'{B}_p0032_1' and x.get('inv') for x in m.get('same') or []), f'нет обратной связи у фото музея: {m.get("same")}'
rows = c.get('/api/rugs').json(); rows = rows if isinstance(rows, list) else (rows.get('items') or rows.get('rugs'))
U = {x['id']: x.get('use') for x in rows}
assert U[f'{B}_p0032_1'] == 'work', U
assert U['azerbaycan_xalchalari_qarabag_qrupu_p0125_1'] == 'maybe', U
shutil.rmtree(tmp, ignore_errors=True)
print('test_books_bundle: OK')
