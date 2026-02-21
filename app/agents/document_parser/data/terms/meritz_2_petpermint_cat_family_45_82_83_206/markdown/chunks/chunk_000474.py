from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA025 NAA026 | 골절 (뒷다리) |\n'
 '| 2 | 눈 및 부속 기관의 질환 | AIA001 | 눈 및 부속 기관의 양성 신생물 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | AIB001 | 눈 및 부속 기관의 악성 신생물 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | AIC001 | 눈 및 부속 기관의 신생물 (양성 또는 악성이 불확실한) |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA001 | 안검 외반 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA002 | 안검 내반 |'),
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
