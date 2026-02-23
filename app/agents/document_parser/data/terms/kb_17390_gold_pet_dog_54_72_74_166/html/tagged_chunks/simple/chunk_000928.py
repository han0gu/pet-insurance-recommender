from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 계약자의 재가입 의사가 확인되었을 때에는 제1항 및 제2항에<br>서 정한 절차에 따라 회사가 재가입 의사를 확인한 날에 '
 '판매중인 제2항의 반려동<br>물보험 상품으로 재가입하는 것으로 하며, 기존 계약은 해지됩니다'),
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
 'indexing': {'chunk_id': 'chunk_000928',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
