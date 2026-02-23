from langchain_core.documents import Document

chunk = Document(
    page_content=('다)의 보험기간이 끝난 날의 다음 날을 말합니다.# ⑦ (재가입형) 특별약관 재가입 관련 용어- 1. 최초계약 : 최초로 체결되는 계약을 '
 '말합니다.\n'
 '- 2. 재가입계약 : 이 보험의 사업방법서에서 정한 재가입 절차에 따라 재가입된 계약을\n'
 '- 말합니다.\n'
 '제2관 보험금의 지급# 제 3조 (보험금의 지급사유)회사는 피보험자가 보험기간 중에 상해로 장해분류표([별표2] 참조. 이하 '
 '같습니다)에서\n'
 '정한 장해지급률이 80% 이상에 해당하는 장해상태가 되었을 때에는 최초 1회에 한하여'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
