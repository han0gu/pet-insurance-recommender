from langchain_core.documents import Document

chunk = Document(
    page_content=('| 2 | 눈 및 부속 기관의 질환 | FAA002 | 안검 내반 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA003 | 안검염 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA004 | 다래끼 / 산립종 / 마이봄선종 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA005 | 체리아이 · 제3안검 돌출 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA006 | 비루관폐쇄 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA007 | 유루증 |'),
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
