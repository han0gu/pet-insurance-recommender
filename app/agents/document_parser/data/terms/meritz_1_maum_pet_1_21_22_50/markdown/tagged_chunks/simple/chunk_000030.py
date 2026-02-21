from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보된 전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함)\n'
 '4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류② 제1항 제2호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 '
 '내용에 따라 국\n'
 '내의 동물병원에서 수의사에 의해 발급한 것이어야 합니다.【수의사법 제12조(진단서 등)】- ① 수의사는 자기가 직접 진료하거나 검안하지 '
 '아니하고는 진단서, 검안서, 증명서\n'
 '- 또는 처방전(「전자서명법」에 따른 전자서명이 기재된 전자문서 형태로 작성한'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000030',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
