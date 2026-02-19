from langchain_core.documents import Document

chunk = Document(
    page_content=('때에는 최고(독촉)기간은 그 다음 날까지로 합니다)으로 정 하여 아래 사항에 대하여 서면(등기우편 등), 전화(음성녹 음) 또는 전자문서 '
 '등으로 알려드립니다. 다만, 해지 전에 발생한 보험금 지급사유에 대하여 회사는 보상합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 73},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 127,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
