from langchain_core.documents import Document

chunk = Document(
    page_content=('무릎뼈 탈구</td></tr><tr><td>NAA024</td><td>십자 인대 손상 파열 (전방 / '
 '후방)</td></tr><tr><td>NAA025 NAA026</td><td>골절 (뒷다리)</td></tr><tr><td '
 'rowspan="13">2</td><td rowspan="13">눈 및 부속 기관의 질환</td><td>AIA001</td><td>눈 및 '
 '부속 기관의 양성 신생물</td></tr><tr><td>AIB001</td><td>눈 및 부속 기관의 악성'),
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
