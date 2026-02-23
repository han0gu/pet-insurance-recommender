from langchain_core.documents import Document

chunk = Document(
    page_content=("id='123' data-category='paragraph' style='font-size:16px'>\uf000 제1항 제2호의 "
 "사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나</p><br><p id='124' "
 "data-category='paragraph' style='font-size:14px'>의원 또는 국외의 의료관련법에서 정한 의료기관에서 "
 '발급한 것이어야 합니다'),
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
 'indexing': {'chunk_id': 'chunk_000612',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
