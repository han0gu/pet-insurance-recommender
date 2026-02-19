from langchain_core.documents import Document

chunk = Document(
    page_content=('【잔존유치】 | 영구치가 났는데도 불구하고 유치가 남아있어서 발치를 하는 경우\n'
 '【잠복고환】 | 고환이 음낭까지 내려오지 못하는 증상\n'
 '제6조(손해의 통지 및 조사)\n'
 '① 계약자 또는 피보험자는 제4조(보상하는 손해)에서 정한 사고가 생긴 것을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental', 'other']},
 'indexing': {'chunk_id': 'chunk_000027',
              'chunk_char_len': 163,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
