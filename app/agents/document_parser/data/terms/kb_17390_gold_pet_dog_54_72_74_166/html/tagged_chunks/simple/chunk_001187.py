from langchain_core.documents import Document

chunk = Document(
    page_content=('지는 사고가 생긴 때에는 피해자는 이 특<br>별약관에 따라 회사가 피보험자에게 지급책임을 지는 금액 한도내에서 회사에 대<br>하여 '
 '보험금의 지급을 직접 청구할 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_001187',
              'chunk_char_len': 97,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
