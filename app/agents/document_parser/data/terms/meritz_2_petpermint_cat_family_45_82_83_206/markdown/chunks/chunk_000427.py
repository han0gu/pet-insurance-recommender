from langchain_core.documents import Document

chunk = Document(
    page_content=('상비율 70% 가입, 연간 첫번째 MRI,CT 및 내시경처\n'
 '치)- ·피보험자가 부담한 치료비 103만원\n'
 '- ·보험금 지급금액\n'
 '= [(103만원 - 3만원)×70%, 50만원] 중 적은금액\n'
 '= 50만원③ 입원 중 MRI,CT 및 내시경처치와 수술을 동시에 한\n'
 '경우(보상비율 70% 가입)- ·피보험자가 부담한 수술당일 치료비 410만원\n'
 '- ·보험금 지급금액\n'
 '- = [(410만원-3만원)×70%, 250만원] 중 적은금액\n'
 '- = 250만원(MRI,CT 및 내시경처치와 수술을 동시에\n'
 '- 하더라도 수술한도로 지급)'),
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
