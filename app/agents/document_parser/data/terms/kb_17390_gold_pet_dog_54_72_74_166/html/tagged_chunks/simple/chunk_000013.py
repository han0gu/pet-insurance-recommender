from langchain_core.documents import Document

chunk = Document(
    page_content=('. 복리 원금 100원, 이자율 연 10%를 가정할 때 - 1년 후 : 100원 + (100원 × 10%) = 110원 - 2년 후 : '
 '110원 + (110원 × 10%) = 121원</td></tr><tr><td>평균공시이율</td><td>전체 보험회사 공시이율의 '
 '평균으로, 이 계약 체결 시점의 이율을 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
