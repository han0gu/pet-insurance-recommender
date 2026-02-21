from langchain_core.documents import Document

chunk = Document(
    page_content=('모두 포함됩니다.나. 질병: 상해를 제외한 상병을 포함합니다. 단, 약관에서 명기하는 보상하지 않는 질병은 제외\n'
 '합니다.# 3. 보상 관련 용어- 가. 보험가입금액: 회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가 지급할 최\n'
 '- 대 보험금을 말합니다.\n'
 '- 나. 자기부담금: 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부담하는 일\n'
 '- 정 금액을 말합니다.\n'
 '- 다. 보험금 분담: 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000005',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
