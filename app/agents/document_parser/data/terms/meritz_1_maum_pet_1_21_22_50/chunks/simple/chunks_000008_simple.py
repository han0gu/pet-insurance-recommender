from langchain_core.documents import Document

chunk = Document(
    page_content=('3. 보상 관련 용어\n'
 '가. 보험가입금액: 회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가 지급할 최대 보험금을 말합니다. 나. 자기부담금: '
 '보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부 담하는 일정 금액을 말합니다. 다. 보험금 분담: 이 계약에서 보장하는 '
 '위험과 같은 위험을 보장하는 다른 계약(공제 계약을 포함합니다)이 있을 경우 비율에 따라 손해를 보상합니다.\n'
 '4. 지급금과 이자율 관련 용어'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 2},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000008',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
