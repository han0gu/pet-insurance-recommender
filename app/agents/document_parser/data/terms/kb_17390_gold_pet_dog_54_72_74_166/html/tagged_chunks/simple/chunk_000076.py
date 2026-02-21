from langchain_core.documents import Document

chunk = Document(
    page_content=('X (1+ 평균공시이율)</td></tr></tbody></table> '
 '<table><thead></thead><tbody><tr><td>※</td><td>2026년 4월 10일 2천만원 X (1+ '
 '평균공시이율)2 평균공시이율이란 전체 보험회사 공시이율의 평균으로, 이 계약 체결 시점 의 이율을 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000076',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
