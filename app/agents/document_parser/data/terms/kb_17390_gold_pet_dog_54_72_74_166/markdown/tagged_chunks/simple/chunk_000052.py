from langchain_core.documents import Document

chunk = Document(
    page_content=('| 예 시 계약자가 2명 이상인 경우 계약자가 2명 이상인 경우 계약전 알릴의무, 보험료 납입의무 등 보험계약에 따른 계약자의 의무를 '
 '연대로 합니다. ∙ 연대 2인 이상이 연대하여 책임을 지므로 각자 채무의 전부를 이행할 책임을 지되 (지분만큼 분할하여 책임을 지는 것과 '
 '다름), 다만 어느 1인의 이행으로 나머 | 예 시 계약자가 2명 이상인 경우 계약자가 2명 이상인 경우 계약전 알릴의무, 보험료 '
 '납입의무 등 보험계약에 따른 계약자의 의무를 연대로 합니다'),
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
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
