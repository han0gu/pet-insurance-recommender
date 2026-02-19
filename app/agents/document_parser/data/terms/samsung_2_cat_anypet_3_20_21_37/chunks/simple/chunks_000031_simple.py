from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 입원만의 경우\n'
 '가. 적용 자기부담금 : 입원 1일당 자기부담금 × 입원일수 나. 적용 지급한도액 : 입원 1일당 지급한도액 × 입원일수\n'
 '2. 입원 중에 수술이 이루어진 경우\n'
 '가. 적용 자기부담금: 입원 1일당 자기부담금 × 입원일수 + 수술 1회당 자기부담금 × 수술횟수 나. 적용 지급한도액: 입원 1일당 '
 '지급한도액 × 입원일수 + 수술 1회당 지급한도액 × 수술횟수\n'
 '③ 통원 또는 통원 당일 수술이 이루어진 경우의 적용 지급한도액은 아래와 같습니다.\n'
 '1. 통원만의 경우\n'
 '가. 적용 자기부담금 : 1일당 자기부담금'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
