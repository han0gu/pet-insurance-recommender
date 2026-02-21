from langchain_core.documents import Document

chunk = Document(
    page_content=('보<br>험금을 지급합니다.<br>\uf000 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위반을 이유로 계약을 해지<br>하거나 '
 '보험금 지급을 거절하지 않습니다.<br>\uf000 보통약관 제1절 일반조항 제29조(보험료의 납입을 연체하여 해지된 계약의 '
 '부활<br>(효력회복))에 따라 이 계약이 부활(효력회복)된 경우에는 부활(효력회복)계약을<br>제2항의 최초계약으로 봅니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
