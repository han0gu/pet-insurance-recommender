from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자가 2명</td><td>이상인 경우에는 그 책임을 연대로 합니다.</td></tr><tr><td colspan="2">예 시 '
 '계약자가 2명 이상인 경우 계약자가 2명 이상인 경우 계약전 알릴의무, 보험료 납입의무 등 보험계약에 따른 계약자의 의무를 연대로 합니다'),
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
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
