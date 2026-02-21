from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약을 취소하는 경우 회<br>사는 최초연장된 날 이후 계약자가 납입한 보험료 전액을 환급합니다.<br>\uf000 제5항에 따라 '
 '보험계약이 연장된 경우 보험계약의 연장일은 회사가 계약자의 재<br>가입의사를 확인한 날(계약자 등이 회사에 보험금을 청구함으로써 '
 '계약자에게 연<br>락이 닿아 회사가 계약자의 재가입의사를 확인한 날 등)까지로 합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000926',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
