from langchain_core.documents import Document

chunk = Document(
    page_content=('해약환급금을 청구하여야 하며, 회사는 청구를 접수한 날부\n'
 '터 3영업일 이내에 해약환급금을 지급합니다. 해약환급금\n'
 '지급일까지의 기간에 대한 이자의 계산은【별표1(보험금을\n'
 '지급할 때의 적립이율 계산)】에 따릅니다.\n'
 '\uf000 회사는 경과기간별 해약환급금에 관한 표를 계약자에게\n'
 '제공하여 드립니다.\n'
 '\uf000 제32조의1(위법계약의 해지)에 따라 위법계약이 해지되\n'
 '는 경우 회사가 적립한 해지 당시의 계약자적립액 및 미경\n'
 '과보험료를 반환하여 드립니다.# 제36조(보험계약대출)\uf000 계약자는 이 계약의 해약환급금 범위 내에서 회사가 정'),
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
