from langchain_core.documents import Document

chunk = Document(
    page_content=('제5항에 관계없이 약정한 보험금을 지급합<br>니다.<br>\uf000 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위<br>반을 '
 '이유로 계약을 해지하거나 보험금 지급을 거절하지 않<br>습니다.<br>\uf000 제30조(보험료의 납입을 연체하여 해지된 계약의 '
 '부활<br>(효력회복))에 따라 이 계약이 부활이 이루어진 경우에는<br>부활계약을 제2항의 최초계약으로 '
 '봅니다.(부활(효력회복)<br>이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을<br>최초계약으로 봅니다)</p><h1 '
 "id='55'"),
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
