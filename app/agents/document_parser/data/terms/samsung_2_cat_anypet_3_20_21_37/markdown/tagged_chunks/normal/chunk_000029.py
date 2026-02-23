from langchain_core.documents import Document

chunk = Document(
    page_content=('나. 적용 지급한도액: 통원 1일당 지급한도액 + 수술 1회당 지급한도액 × 수술횟수# 제10조(보험금의 분담)① 이 계약에서 보장하는 '
 '위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경\n'
 '우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상책임액의 합계액이 손해액을\n'
 '초과할 때에는 회사는 아래에 따라 손해를 보상합니다.| 다른 계약이 없을 때 이 계약의 보상책임액 손해액(피보험자가 부담한 총비용) × '
 '다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액 |\n'
 '| --- |'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
