from langchain_core.documents import Document

chunk = Document(
    page_content=('- 된 경우에는 해당기간에 대하여 가산이율을 적\n'
 '199# 용하지 않습니다.- 5. 가산이율 적용시 금융위원회 또는 금융감독원이\n'
 '- 정당한 사유로 인정하는 경우에는 해당 기간에\n'
 '- 대하여 가산이율을 적용하지 않습니다.\n'
 '- 6. 회사가 해지권을 행사하는 경우 위 표의 ‘청구\n'
 '- 일’은 보험사의 해지 의사표시(서면, 전자우\n'
 '- 편, 휴대전화 문자메시지 또는 이에 준하는 전\n'
 '- 자적 의사표시 포함)가 보험계약자 또는 그의\n'
 '- 대리인에게 도달한 날로 봅니다.'),
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
