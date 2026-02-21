from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>\uf000 회사가 제1조(보험금의</p><br><p id='168' "
 "data-category='paragraph' style='font-size:14px'>지급사유)에서 정한 2대호흡계특정질환진단비를 "
 "지급한</p><p id='169' data-category='paragraph' style='font-size:16px'>- 90 "
 "-</p><p id='170' data-category='paragraph' style='font-size:16px'>경우에는 그 "
 '지급사유가 발생한 때부터 이 특별약관'),
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
 'indexing': {'chunk_id': 'chunk_000637',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
