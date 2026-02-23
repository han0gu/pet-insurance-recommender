from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 제13조(손해배상청구에 대한 회사의 해결) 제2항 및 제3항의<br>제6조(손해의 발생과 통지)<br>회사의 요구에 따르기 '
 '위하여 지출한 비용 반<br>\uf000 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체없이 그 내용을<br>회사에 알려야 '
 '합니다. 려동<br>제5조(보상하지 않는 손해)<br>1'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001155',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
