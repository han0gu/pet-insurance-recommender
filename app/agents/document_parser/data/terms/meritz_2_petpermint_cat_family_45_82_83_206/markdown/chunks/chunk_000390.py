from langchain_core.documents import Document

chunk = Document(
    page_content=('= 30만원③ 통원 중 MRI,CT 및 내시경처치와 수술을 동시에 한\n'
 '경우(보상비율 50% 가입)- ·피보험자가 부담한 수술당일 치료비 410만원\n'
 '- ·보험금 지급금액\n'
 '- = [(410만원-3만원)×50%, 200만원] 중 적은금액\n'
 '- = 200만원(MRI,CT 및 내시경처치와 수술을 동시에\n'
 '- 하더라도 수술한도로 지급)\n'
 '\uf000 수술과 MRI,CT 및 내시경처치를 동일한 날에 시행한 경\n'
 '우 수술한 날의 지급한도 내에서 보험금이 지급됩니다.\n'
 '\uf000 연간 1년 이내에 각각 다른 MRI,CT 및 내시경처치를 받'),
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
