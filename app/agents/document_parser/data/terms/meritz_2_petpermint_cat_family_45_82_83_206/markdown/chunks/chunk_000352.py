from langchain_core.documents import Document

chunk = Document(
    page_content=('- 발생일로부터 과거 1년 이내의 예방접종 기록이 있는\n'
 '- 경우에는 보상합니다.)\n'
 '- : 파보 바이러스 감염, 디스템퍼 바이러스 감염, 파라\n'
 '139인플루엔자 감염, 전염성 간염, 아데노 바이러스 2\n'
 '형 감염, 광견병, 코로나 바이러스 감염, 렙토스피\n'
 '라 감염, 필라리아(심장사상충) 감염, 인플루엔자\n'
 '감염, 고양이범백혈구감소증, 고양이칼리시바이러스\n'
 '감염증, 고양이바이러스성비기관지염, 고양이백혈병\n'
 '바이러스감염증, 고양이헤르페스바이러스감염증, 고\n'
 '양이클라미디아감염증- ③ 상병명을 알 수 없는 상해 또는 질병에 대한 치료'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
