from langchain_core.documents import Document

chunk = Document(
    page_content=('항목 | 자기부담금 | 지급 한도\n'
 '통원 또는 입원하는 경우 | 1일당 ( )원 | 1일당 ( )원 / 연간 ( )원\n'
 '【보험금 지급금액 예시】아래의 경우는 이해를 돕기 위한 예시이며, 자기부담금, 지급한도 등은 달라질 수 있습니다.\n'
 '- 보험계약일(보장개시일) : 2025년 5월 1일 - 자기부담금: 3만원 - 보상비율: 70% - 1일 보상한도: 20만원 - 연간 '
 '총보상한도: 30만원\n'
 '입˙통원 진료일 | 2025.5.1 | 2025.8.1. | 2025.11.1\n'
 '피보험자가 부담한 치료비 | 10만원 | 20만원 | 40만원'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 3},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000015',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
