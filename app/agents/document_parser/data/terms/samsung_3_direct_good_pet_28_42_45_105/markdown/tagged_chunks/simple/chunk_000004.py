from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 연단위 복리: 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를\n'
 '원금에 더한 금액을 다음 1년의 원금으로 하는 이자 계산방법을 말합니다.# <예시안내># [연단위 복리]원금 100원, 연간 10% '
 '이자율 적용시 연단위 복리로 계산한 2년 시점의 총 이자 금액- · 1년차 이자 = 100원(※원금) ×10% = 10원\n'
 '- · 2년차 이자 = (100원 +10원)(※원금+1년차 이자) ×10% = 11원\n'
 '- → 2년 시점의 총 이자금액 = 10원 + 11원 = 21원'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000004',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
