from langchain_core.documents import Document

chunk = Document(
    page_content=('원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.# <예시안내># [연단위 복리]원금 100원, 연간 10% '
 '이자율 적용시 연단위 복리로 계산한 2년 시점의 총 이자 금액- ∙ 1년차 이자 = 100원(※원금) ×10% = 10원\n'
 '- ∙ 2년차 이자 = (100원 + 10원)(※원금+1년차 이자) ×10% = 11원\n'
 '- → 2년 시점의 총 이자금액 = 10원 + 11원 = 21원\n'
 '- 2. 평균공시이율: 전체 보험회사 공시이율의 평균으로, 이 계약 체결 시점의 이율을'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000004',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
