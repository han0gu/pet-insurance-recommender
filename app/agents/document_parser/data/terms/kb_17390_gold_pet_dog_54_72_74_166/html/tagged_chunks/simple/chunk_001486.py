from langchain_core.documents import Document

chunk = Document(
    page_content='‘안구(눈동자)의 뚜렷한 조절기능장해’라 함은 조절력이 정상의 1/2<br>이하로 감소된 경우를 말한다',
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001486',
              'chunk_char_len': 57,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
