from langchain_core.documents import Document

chunk = Document(
    page_content=("- 관한 규정'에 따른 공휴일과 노동절을 제외합니다.\n"
 '# 제3조(보험목적의 범위)- ① 이 약관에서 보험의 목적이라 함은 이 약관에 따라 보험에 가입한 반려동물로 보험증권에 기재된\n'
 '- 반려견을 말합니다.\n'
 '- ② 이 약관에서 반려동물이라 함은 아래에 해당하는 반려견을 말합니다.\n'
 '- 1. 동물보호법 시행령 제4조 제1호에 의거하여 주택·준주택에서 기르는 개(犬)\n'
 '# 제2관 보험금의 지급# 제4조(보상하는 손해)- ① 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 상해 또는 질병(이하 '
 '"사고"라 합니다)이'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
