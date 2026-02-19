from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 지급금과 이자율 관련 용어\n'
 '가. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로 하는 '
 '이자 계산방법을 말합니다. 나. 보험개발원이 공시하는 보험계약대출이율: 보험개발원이 정기적으로 산출하여 공시 하는 이율로써 회사가 '
 '보험금의 지급 또는 보험료의 환급을 지연하는 경우 등에 적 용합니다.\n'
 '5. 기간과 날짜 관련 용어'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 2},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
