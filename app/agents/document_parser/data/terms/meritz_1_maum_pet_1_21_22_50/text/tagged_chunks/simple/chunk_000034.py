from langchain_core.documents import Document

chunk = Document(
    page_content=('화 문자메시지 또는 전자우편 등으로도 송부하며, 그 서류를 접수한 날부터 3영업일 이\n'
 '내에 보험금을 지급합니다.\n'
 '② 회사가 보험금 지급사유를 조사·확인하기 위해 필요한 기간이 제1항의 지급기일을 초과\n'
 '할 것이 명백히 예상되는 경우에는 그 구체적인 사유와 지급예정일 및 보험금 가지급제\n'
 '도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여 피보험자 또는 보험수익자에\n'
 '게 즉시 통지합니다. 다만, 지급예정일은 다음 각 호의 어느 하나에 해당하는 경우를'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000034',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
