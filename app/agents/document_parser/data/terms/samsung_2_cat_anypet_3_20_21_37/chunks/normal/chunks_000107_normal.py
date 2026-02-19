from langchain_core.documents import Document

chunk = Document(
    page_content=('【현저하게 공정을 잃은 합의】 회사가 계약자 또는 피보험자의 경제적. 신체적. 정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 '
 '동일·유사 사례에 비추어 계약자 또는 피보험자에게 매우 불합리하게 합의를 하 는 것을 의미합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 127,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
