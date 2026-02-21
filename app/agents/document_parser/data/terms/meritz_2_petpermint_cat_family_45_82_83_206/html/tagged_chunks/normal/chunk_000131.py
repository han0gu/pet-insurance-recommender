from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 계약자가 청약할 때에 계약자에게 약관의 중요한<br>내용을 설명하여야 하며, 청약 후에 다음 각 호의 방법 중<br>계약자가 '
 '원하는 방법을 확인하여 지체 없이 약관 및 계약<br>자 보관용 청약서를 제공하여 드립니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 124,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
