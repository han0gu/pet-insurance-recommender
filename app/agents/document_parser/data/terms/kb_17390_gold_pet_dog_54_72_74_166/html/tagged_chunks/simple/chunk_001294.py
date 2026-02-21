from langchain_core.documents import Document

chunk = Document(
    page_content=('. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인<br>이 아닌 경우에는 본인의 인감증명서, '
 '본인서명사실확인서 또는 안전성과 신<br>뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포함)<br>6'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001294',
              'chunk_char_len': 136,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
