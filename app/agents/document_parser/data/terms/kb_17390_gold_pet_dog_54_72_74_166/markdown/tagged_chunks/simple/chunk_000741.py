from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신\n'
 '- 뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)\n'
 '- 6. 수탁기관 위탁비용 영수증 및 동물관리위탁업자가 제공하는 계약서(위탁관리\n'
 '- 업소 등록번호, 업소명 및 주소, 전화번호, 위탁관리동물 종류, 품종, 나이,\n'
 '- 서비스 기간, 비용 등 포함)\n'
 '7. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류\n'
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
 'indexing': {'chunk_id': 'chunk_000741',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
