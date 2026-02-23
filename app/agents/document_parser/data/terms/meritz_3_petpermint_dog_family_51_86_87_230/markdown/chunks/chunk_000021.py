from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 선박에 탑승하는 것을 직무로 하는 사람이 직무상 선\n'
 '- 박에 탑승하고 있는 동안\n'
 '# 제6조(보험금 지급사유의 통지)계약자 또는 피보험자나 보험수익자는 제3조(보험금의 지급\n'
 '사유)에서 정한 보험금 지급사유의 발생을 안 때에는 지체\n'
 '없이 그 사실을 회사에 알려야 합니다.# 제7조(보험금의 청구)\uf000 보험수익자는 다음의 서류를 제출하고 보험금을 청구하\n'
 '여야 합니다.- ① 청구서(회사양식)\n'
 '- ② 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단\n'
 '- 서, 입원치료확인서, 의사처방전(처방조제비) 등)'),
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
