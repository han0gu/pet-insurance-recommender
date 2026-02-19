from langchain_core.documents import Document

chunk = Document(
    page_content=('나. 적용 지급한도액 : 통원 1일당 지급한도액\n'
 '2. 통원 당일 수술이 이루어진 경우\n'
 '가. 적용 자기부담금: 통원 1일당 자기부담금 + 수술 1회당 자기부담금 × 수술횟수 나. 적용 지급한도액: 통원 1일당 지급한도액 + '
 '수술 1회당 지급한도액 × 수술횟수\n'
 '제10조(보험금의 분담)\n'
 '① 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경 우 각 계약에 대하여 다른 계약이 없는 '
 '것으로 하여 각각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
