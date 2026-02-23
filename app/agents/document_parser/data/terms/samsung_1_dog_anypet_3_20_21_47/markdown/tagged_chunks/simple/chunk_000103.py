from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 곱한 금액이며 보험증권에 정한 1일당 보상한도액을 한도로 보상합니다.\n'
 '- ② 보험기간 중에 발생한 사고로 회사가 지급하는 슬관절 수술비용보험금의 총 한도는 보험증권에 기\n'
 '- 재된 총보장횟수를 한도로 합니다.\n'
 '【슬관절 수술비용보험금】 아래 ①과 ② 중 적은 금액- ① 피보험자가 부담한 수술 당일치료비 × 보상비율\n'
 '- ② 보험증권에서 정한 1일당 보상한도액'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000103',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
