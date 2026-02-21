from langchain_core.documents import Document

chunk = Document(
    page_content=("107</p><br><p id='88' data-category='paragraph' "
 "style='font-size:14px'>제</p><br><p id='89' data-category='paragraph' "
 "style='font-size:14px'>도</p><h1 id='90' style='font-size:14px'>제20조(중대사유로 인한 "
 "해지)</h1><br><p id='91' data-category='list' style='font-size:14px'>\uf000 "
 '회사는 아래와 같은 사실이 있을 경우에는 안 날부터 1개월 이내에'),
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
 'indexing': {'chunk_id': 'chunk_000910',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
