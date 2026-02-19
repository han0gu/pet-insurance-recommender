from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사는 자동대출납입이 종료된 날부터 15일 이내에 자동 대출납입이 종료되었음을 서면, 전화(음성녹음) 또는 전자 '
 '문서(SMS 포함) 등으로 계약자에게 안내하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 72},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000127',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
