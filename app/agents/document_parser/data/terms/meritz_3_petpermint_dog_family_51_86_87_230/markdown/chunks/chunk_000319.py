from langchain_core.documents import Document

chunk = Document(
    page_content=('# 【보험금 지급금액 산출방식】보험금 지급금액 = [(피보험자가 부담한 치료비－자기부담금)\n'
 '× 보상비율, 지급 한도액] 중 적은 금액【보험금 지급금액(자기부담금 3만원인 경우)[예시]】① 통원 중 수술을 하지 않은 경우(보상비율 '
 '50%)- ·피보험자가 부담한 치료비 13만원\n'
 '- ·보험금 지급금액\n'
 '= [(13만원 - 3만원)×50%, 10만원] 중 적은금액\n'
 '= 5만원② 통원 중 수술을 한 경우(보상비율 50%)- ·피보험자가 부담한 수술당일 치료비 410만원\n'
 '- ·보험금 지급금액'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
