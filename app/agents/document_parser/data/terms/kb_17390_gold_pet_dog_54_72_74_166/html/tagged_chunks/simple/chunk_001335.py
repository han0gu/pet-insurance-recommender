from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자에게 지급합니다.<br>\uf000 이 특별약관의 보험금을 지급하였을 때에는 지급한 보험금액에 해당하는 계약의 '
 '보<br>험가입금액이 지급일에 감액된 것으로 봅니다'),
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
 'indexing': {'chunk_id': 'chunk_001335',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
