from langchain_core.documents import Document

chunk = Document(
    page_content=('내에서 정합니다.- ① 소송제기\n'
 '- ② 분쟁조정 신청\n'
 '- ③ 수사기관의 조사\n'
 '- ④ 해외에서 발생한 보험사고에 대한 조사\n'
 '- ⑤ 제5항에 따른 회사의 조사요청에 대한 동의 거부 등\n'
 '- 계약자, 피보험자 또는 보험수익자의 책임있는 사유로\n'
 '- 보험금 지급사유의 조사와 확인이 지연되는 경우\n'
 '- ⑥ 제7항에 따라 보험금 지급사유에 대해 제3자의 의견에\n'
 '- 따르기로 한 경우\n'
 '# 【분쟁조정 신청】분쟁조정 신청은 이 약관의「분쟁의 조정」조항에 따르\n'
 '며 분쟁조정 신청 대상기관은 금융감독원의 금융분쟁조'),
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
