from langchain_core.documents import Document

chunk = Document(
    page_content=('지급할 때의 적립이율 계산)】에 따릅니다.\n'
 '\uf000 회사는 경과기간별 해약환급금에 관한 표를 계약자에게\n'
 '제공하여 드립니다.\n'
 '\uf000 제32조의1(위법계약의 해지)에 따라 위법계약이 해지되\n'
 '는 경우 회사가 적립한 해지 당시의 계약자적립액 및 미경\n'
 '과보험료를 반환하여 드립니다.# 제36조(보험계약대출)\uf000 계약자는 이 계약의 해약환급금 범위 내에서 회사가 정\n'
 '한 방법에 따라 대출(이하「보험계약대출」이라 합니다)을\n'
 '받을 수 있습니다. 그러나, 순수보장성보험 등 보험상품의\n'
 '종류에 따라 보험계약대출이 제한될 수도 있습니다.'),
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
