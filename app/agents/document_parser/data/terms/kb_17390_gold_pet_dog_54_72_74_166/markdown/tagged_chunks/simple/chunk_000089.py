from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약 체결 전에 피보험자의 | 유 의 사 항 보험계약을 청약하면서 보험설계사에게 질병이 있다고만 얘기하였을 뿐, 청약 공 서의 계약전 '
 '알릴 사항에 아무런 기재도 하지 않을 경우에는 보험설계사에게 통 병력을 얘기하였다고 하더라도 회사는 계약 전 알릴 의무 위반을 이유로 '
 '계약 사항 을 해지하고 보험금을 지급하지 않을 수 있습니다. 계약 체결 전에 피보험자의 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
