from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 손해배상책임의 전부 또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거나 소\n'
 '송, 중재 또는 조정을 제기하거나 신청하고자 할 경우에는 미리 회사의 동의를 받\n'
 '는 일② 계약자 또는 피보험자가 정당한 이유 없이 제1항의 의무를 이행하지 않았을 때에는 제\n'
 '3조(보상하는 손해)에 의한 손해에서 다음의 금액을 뺍니다.1. 제1항 제1호의 경우에는 그 노력을 하였더라면 손해를 방지 또는 경감할 '
 '수 있었던\n'
 '금액\n'
 '2. 제1항 제2호의 경우에는 제3자로부터 손해의 배상을 받을 수 있었던 금액'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
