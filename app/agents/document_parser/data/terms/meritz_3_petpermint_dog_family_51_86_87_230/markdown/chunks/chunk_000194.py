from langchain_core.documents import Document

chunk = Document(
    page_content=('에 대한 신뢰성을 갖춘 전자문서를 포함)으로 동의하여야\n'
 '합니다.\n'
 '\uf000 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계\n'
 '약자에게 보험증권 및 약관을 교부하고 변경된 계약자가 요\n'
 '청하는 경우 약관의 중요한 내용을 설명하여 드립니다.# 제14조(보험나이 등)\uf000 이 특별약관에서의 피보험자 및 반려동물의 나이는 '
 '만나\n'
 '이를 기준으로 합니다.\n'
 '\uf000 제1항의 만나이는 계약일 현재 피보험자 및 반려동물의\n'
 '실제 만나이를 기준으로 하며, 이후 매년 계약해당일에 나\n'
 '이가 증가하는 것으로 합니다.'),
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
