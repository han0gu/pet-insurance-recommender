from langchain_core.documents import Document

chunk = Document(
    page_content=('형 감염, 광견병, 코로나 바이러스 감염, 렙토스피\n'
 '라 감염, 필라리아(심장사상충) 감염, 인플루엔자\n'
 '감염, 고양이범백혈구감소증, 고양이칼리시바이러스\n'
 '감염증, 고양이바이러스성비기관지염, 고양이백혈병\n'
 '바이러스감염증, 고양이헤르페스바이러스감염증, 고\n'
 '양이클라미디아감염증- ③ 상병명을 알 수 없는 상해 또는 질병에 대한 치료\n'
 '- ④ 백신 접종비용 및 기타 질병예방을 위한 검사 또는\n'
 '- 투약·예방 접종비용 및 정기검진, 예방적 검사를\n'
 '- 위한 비용\n'
 '- ⑤ 반려동물의 임신·출산, 제왕절개, 인공유산, 발정과'),
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
