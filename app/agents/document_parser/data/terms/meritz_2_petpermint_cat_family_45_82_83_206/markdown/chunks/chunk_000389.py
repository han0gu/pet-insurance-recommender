from langchain_core.documents import Document

chunk = Document(
    page_content=('MRI,CT 및 내시경처치를 받지 않은 날)- ·피보험자가 부담한 치료비 13만원\n'
 '- ·보험금 지급금액\n'
 '= [(13만원 - 3만원)×50%, 10만원] 중 적은금액\n'
 '= 5만원② 통원 중 MRI,CT 및 내시경처치를 받은 날의 경우(보\n'
 '상비율 50% 가입, 연간 첫번째 MRI,CT 및 내시경처\n'
 '치)- ·피보험자가 부담한 치료비 103만원\n'
 '- ·보험금 지급금액\n'
 '= [(103만원 - 3만원)×50%, 30만원] 중 적은금액\n'
 '= 30만원③ 통원 중 MRI,CT 및 내시경처치와 수술을 동시에 한'),
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
