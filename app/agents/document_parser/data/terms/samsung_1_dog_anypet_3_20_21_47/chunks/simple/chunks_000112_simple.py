from langchain_core.documents import Document

chunk = Document(
    page_content=('【현저하게 공정을 잃은 합의】 회사가 계약자 또는 피보험자의 경제적. 신체적. 정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 '
 '동일·유사 사례에 비추어 계약자 또는 피보험자에게 매우 불합리하게 합의를 하 는 것을 의미합니다.\n'
 '제37조(개인정보보호)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 20},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000112',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
