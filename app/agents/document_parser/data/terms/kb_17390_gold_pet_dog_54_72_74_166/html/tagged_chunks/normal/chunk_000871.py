from langchain_core.documents import Document

chunk = Document(
    page_content=("id='48' data-category='list' style='font-size:14px'>\uf000 회사는 계약의 청약을 승낙하고 "
 '제1회 보험료 등을 받은 때부터 이 특별약관이 정<br>한 바에 따라 보장을 합니다'),
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
 'indexing': {'chunk_id': 'chunk_000871',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
