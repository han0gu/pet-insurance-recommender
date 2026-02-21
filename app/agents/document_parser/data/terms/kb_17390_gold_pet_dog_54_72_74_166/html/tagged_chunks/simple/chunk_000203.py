from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>일정액을 회사가 적립해 둔 금액을 말합니다.</p><p id='16' "
 "data-category='list' style='font-size:16px'>제23조(보험나이 등)<br>\uf000 이 약관에서의 "
 '피보험자의 나이는 보험나이를 기준으로 합니다'),
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
 'indexing': {'chunk_id': 'chunk_000203',
              'chunk_char_len': 156,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
