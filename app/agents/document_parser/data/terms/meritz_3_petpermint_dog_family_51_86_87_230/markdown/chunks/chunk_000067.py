from langchain_core.documents import Document

chunk = Document(
    page_content=('계약의 청약일 이후 5년)이 지나는 동안 보장이 제외되는\n'
 '질병으로 추가 진단(단순 건강검진 제외) 또는 치료사실이\n'
 '없을 경우, 청약일로부터 5년이 지난 이후에는 이 약관에\n'
 '따라 보장합니다.\n'
 '\uf000 제5항의「청약일로부터 5년이 지나는 동안」이라 함은\n'
 '제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)와\n'
 '계약의 해지)에서 정한 계약의 해지가 발생하지 않은 경우\n'
 '를 말합니다.\n'
 '\uf000 제30조(보험료의 납입을 연체하여 해지된 계약의 부활\n'
 '(효력회복))에서 정한 계약의 부활이 이루어진 경우 부활을'),
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
