from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 재가입 적용대상 특별약관이 다음 각 호의 조건을 충족\n'
 '하고 계약자가 제5항에 따라 재가입 의사를 표시한 때에는\n'
 '이 특별약관의 제11조(보험계약의 성립) 및 보통약관 제21\n'
 '조(약관 교부 및 설명 의무 등)를 준용하여 회사가 정한 절\n'
 '차에 따라 계약자는 기존 계약에 이어 재가입할 수 있으며,\n'
 '이 경우 회사는 기존 계약의 가입 이후 발생한 상해 또는\n'
 '질병을 사유로 가입을 거절할 수 없습니다.- ① 재가입일에 있어서 반려동물의 나이가 회사가 최초가\n'
 '- 입 당시 정한 재가입 나이의 범위 내일 것'),
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
