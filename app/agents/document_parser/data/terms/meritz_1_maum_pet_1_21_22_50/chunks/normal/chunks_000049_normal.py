from langchain_core.documents import Document

chunk = Document(
    page_content=('【보험개발원이 공시하는 월평균 정기예금이율】\n'
 '현재 시점의 정기예금이율은 보험개발원 홈페이지(www.kidi.or.kr)에서 확인할 수 있 습니다.\n'
 '【연단위 복리】\n'
 '회사가 지급할 금전에 이자를 줄 때, 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 '
 '말합니다. 원금 100원, 이자율 연 10%를 가정할 때\n'
 '- 1년 후 원리금 : 100원 + (100원×10%) = 110원 - 2년 후 원리금 : 110원 + (110원×10%) = 121원\n'
 '제12조(주소변경통지)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000049',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
