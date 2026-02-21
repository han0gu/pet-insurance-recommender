from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑤ 제6항에 따른 회사의 조사요청에 대한 동의 거부 등\n'
 '- 계약자, 피보험자 또는 보험수익자의 책임있는 사유로\n'
 '- 인하여 보험금 지급사유의 조사 및 확인이 지연되는\n'
 '- 경우\n'
 '- ⑥ 보험금 지급사유에 대해 제3자의 의견에 따르기로 한\n'
 '- 경우\n'
 '# 【분쟁조정 신청】분쟁조정 신청은 이 약관의「분쟁의 조정」조항에 따르\n'
 '며 분쟁조정 신청 대상기관은 금융감독원의 금융분쟁조\n'
 '정위원회를 말합니다.\uf000 제2항에 따라 장해지급률의 판정 및 지급할 보험금의 결\n'
 '정과 관련하여 확정된 장해지급률에 따른 보험금을 초과한'),
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
