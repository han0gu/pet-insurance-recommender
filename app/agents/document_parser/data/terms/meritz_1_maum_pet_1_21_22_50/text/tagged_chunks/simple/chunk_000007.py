from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 보험금 분담: 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제\n'
 '계약을 포함합니다)이 있을 경우 비율에 따라 손해를 보상합니다.4. 지급금과 이자율 관련 용어가. 연단위 복리: 회사가 지급할 금전에 '
 '이자를 줄 때 1년마다 마지막 날에 그 이자를\n'
 '원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.\n'
 '나. 보험개발원이 공시하는 보험계약대출이율: 보험개발원이 정기적으로 산출하여 공시\n'
 '하는 이율로써 회사가 보험금의 지급 또는 보험료의 환급을 지연하는 경우 등에 적'),
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
 'indexing': {'chunk_id': 'chunk_000007',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
