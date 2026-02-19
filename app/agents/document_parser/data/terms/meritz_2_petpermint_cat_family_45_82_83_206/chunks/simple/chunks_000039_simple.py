from langchain_core.documents import Document

chunk = Document(
    page_content=('【운용자산이익률】\n'
 '직전 1년간의 운용자산에 대한 투자영업수익과 투자영 업비용 등을 고려하여 산출\n'
 '【외부지표금리】\n'
 '국고채, 회사채, 통화안정증권, 양도성예금증서 등을 고려하여 산출\n'
 '제10조(만기환급금의 지급)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 55},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000039',
              'chunk_char_len': 115,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
