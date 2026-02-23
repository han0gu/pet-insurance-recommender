from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 청구서(회사 양식)\n'
 '- 2. 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단서, 입원치료확인서,\n'
 '- 의사처방전(처방조제비) 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인\n'
 '- 이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰\n'
 '- 성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류\n'
 '\uf000 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000021',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
