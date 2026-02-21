from langchain_core.documents import Document

chunk = Document(
    page_content=('지급사유)에서 장해지급률이 상해 발생일부터 180일 이내에 확정<br>되지 않는 경우에는 상해 발생일부터 180일이 되는 날의 의사 '
 '진단에 기초하여 고<br>정될 것으로 인정되는 상태를 장해지급률로 결정합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000337',
              'chunk_char_len': 116,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
