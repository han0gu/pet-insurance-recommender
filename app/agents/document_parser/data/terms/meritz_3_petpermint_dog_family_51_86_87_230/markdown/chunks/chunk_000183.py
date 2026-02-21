from langchain_core.documents import Document

chunk = Document(
    page_content=('반을 이유로 계약을 해지하거나 보험금 지급을 거절하지 않\n'
 '습니다.\n'
 '\uf000 보통약관 제30조(보험료의 납입을 연체하여 해지된 계약\n'
 '의 부활(효력회복))에 따라 이 계약이 부활이 이루어진 경\n'
 '우에는 부활계약을 제2항의 최초계약으로 봅니다.(부활(효\n'
 '력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회\n'
 '복)계약을 최초계약으로 봅니다)# 제10조(사기에 의한 계약)\uf000 계약자 또는 피보험자가 사기에 의하여 계약이 성립되었\n'
 '음을 회사가 증명하는 경우에는 계약일부터 5년 이내(사기\n'
 '사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니'),
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
