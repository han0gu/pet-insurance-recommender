from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 다른 보험가입내역에 대한 계약 전 알릴 의무 위반을 이유로 계약을 해지\n'
 '- 하거나 보험금 지급을 거절하지 않습니다.\n'
 '- \uf000 제29조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에 따라 이 계약\n'
 '- 이 부활(효력회복)된 경우에는 부활(효력회복)계약을 제2항의 최초계약으로 봅니\n'
 '- 다. 또한, 부활(효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계\n'
 '- 약을 최초계약으로 봅니다.\n'
 '|  |  |\n'
 '| --- | --- |'),
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
