from langchain_core.documents import Document

chunk = Document(
    page_content=('경우에 해당되는 사유로 보험계<br>약에서 정한 보험금의 지급사유가 발생한 경우에는 보험금을 지급합니다.</p><br><p '
 "id='123' data-category='list' style='font-size:16px'>1"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001453',
              'chunk_char_len': 123,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
