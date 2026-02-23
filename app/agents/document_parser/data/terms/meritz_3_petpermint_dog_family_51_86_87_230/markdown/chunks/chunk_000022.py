from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서, 입원치료확인서, 의사처방전(처방조제비) 등)\n'
 '- ③ 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정\n'
 '- 부기관발행 신분증, 본인이 아닌 경우에는 본인의 인\n'
 '- 감증명서, 본인서명사실확인서 또는 안전성과 신뢰성\n'
 '- 이 확보된 전자적 수단을 활용한 보험수익자 의사표시\n'
 '- 의 확인방법 포함)\n'
 '- ④ 기타 보험수익자가 보험금의 수령에 필요하여 제출하\n'
 '- 는 서류\n'
 '\uf000 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에\n'
 '서 규정한 국내의 병원이나 의원 또는 국외의 의료관련법에'),
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
