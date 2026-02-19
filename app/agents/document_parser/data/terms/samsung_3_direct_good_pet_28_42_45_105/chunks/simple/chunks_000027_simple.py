from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의 원 또는 국외의 의료관련법에서 정한 의료기관에서 '
 '발급한 것이어야 합니다.\n'
 '<관련법규>\n'
 '[의료법 제3조(의료기관)]\n'
 '제8조 (보험금의 지급절차)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 30},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000027',
              'chunk_char_len': 128,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
