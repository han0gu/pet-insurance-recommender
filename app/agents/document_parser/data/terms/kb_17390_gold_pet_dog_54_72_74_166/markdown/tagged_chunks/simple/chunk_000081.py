from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이행하지 않았을때\n'
 '- 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계약\n'
 '\uf000을 해지할 수 없습니다.\n'
 '1. 회사가 최초 계약 체결 당시에 그 사실을 알았거나 과실로 인하여 알지 못하였- 을 때\n'
 '- 2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 또는 제1회 보험료를 받은 때\n'
 '- 부터 보험금 지급사유가 발생하지 않고 2년(진단계약의 경우 질병에 대하여는\n'
 '- 1년)이 지났을 때\n'
 '- 3. 최초 계약을 체결한 날부터 3년이 지났을 때'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
