from langchain_core.documents import Document

chunk = Document(
    page_content=("id='27' data-category='paragraph' style='font-size:16px'>제4조(갱신보장특약 제1회 보험료의 "
 "납입최고(독촉)와 계약의 해지)</p><br><p id='28' data-category='paragraph' "
 "style='font-size:16px'>\uf000 계약자는 보통약관 제1절 일반조항 제28조(보험료의 납입이 연체되는 "
 "경우</p><br><p id='29' data-category='paragraph' "
 "style='font-size:14px'>반</p><br><p id='30'"),
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
 'indexing': {'chunk_id': 'chunk_001386',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
