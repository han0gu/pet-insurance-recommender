from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다)의 청약과 보험회사의 승낙으로 보험계약(이하 「계\n'
 '약」이라 합니다)에 부가하여 이루어집니다.\n'
 '\uf000 제1항에 따라 이 특약을 부가할 때 반려동물의 과거 병\n'
 '력과 수의학적으로 또는 경험통계적으로 인과관계가 유의성\n'
 '있게 확인된 경우 등과 같이 회사가 정한 기준에 따라 직접\n'
 '관련이 있는 특정질병으로 제한하며, 부담보 설정 범위 및\n'
 '사유를 계약자에게 설명하여 드립니다.\n'
 '\uf000 이 특별약관의 보장개시일은 보통약관 제26조(제1회 보\n'
 '험료 및 회사의 보장개시)에서 정한 보장개시일과 동일합니\n'
 '다.'),
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
