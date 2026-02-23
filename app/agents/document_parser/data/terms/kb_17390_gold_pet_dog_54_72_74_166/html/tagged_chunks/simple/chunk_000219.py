from langchain_core.documents import Document

chunk = Document(
    page_content=("65</p><br><p id='29' data-category='paragraph' style='font-size:20px'>- 65 "
 "-</p><p id='30' data-category='paragraph' style='font-size:20px'>제 5 관 보험료의 "
 "납입</p><p id='31' data-category='list' style='font-size:14px'>제 25조(제1회 보험료 및 "
 '회사의 보장개시)<br>\uf000 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 바에<br>따라 보장을'),
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
 'indexing': {'chunk_id': 'chunk_000219',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
