from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이\n'
 '- "금융소비자보호에 관한 법률" 제42조에서 정하는 일정 금액 이내인 분쟁사건에\n'
 '- 대하여 조정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외하고는 소를\n'
 '- 제기하지 않습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000200',
              'chunk_char_len': 150,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
