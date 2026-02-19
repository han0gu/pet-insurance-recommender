from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(보험목적의 범위)\n'
 '① 이 약관에서 보험의 목적이라 함은 이 약관에 따라 보험에 가입한 반려동물로 보험증권에 기재된 반려묘을 말합니다. ② 이 약관에서 '
 '반려동물이라 함은 아래에 해당하는 반려묘을 말합니다. 1. 주택법 제2조 제1호 및 제4호에 의거하여 주택·준주택에서 기르는 고양이\n'
 '제2관 보험금의 지급\n'
 '제4조(보상하는 손해)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 186,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
