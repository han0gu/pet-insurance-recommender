from langchain_core.documents import Document

chunk = Document(
    page_content=('자가 위반사항을 안 날로부터 1년 이내에 계약해지요구서에\n'
 '증빙서류를 첨부하여 위법계약의 해지를 요구할 수 있습니\n'
 '다.\n'
 '\uf000 회사는 해지요구를 받은 날부터 10일 이내 수락여부를\n'
 '계약자에 통지하여야 하며, 거절할 때에는 거절 사유를 함\n'
 '께 통지하여야 합니다.\n'
 '\uf000 계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르\n'
 '지 않는 경우 해당 계약을 해지할 수 있습니다.\n'
 '\uf000 제1항 및 제3항에 따라 계약이 해지된 경우 회사는 제35\n'
 '조(해약환급금) 제4항에 따른 해약환급금을 계약자에게 지\n'
 '급합니다.'),
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
