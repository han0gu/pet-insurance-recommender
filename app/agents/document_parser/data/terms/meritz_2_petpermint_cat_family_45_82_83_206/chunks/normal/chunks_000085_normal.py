from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자는 서면 등을 발송한 때에 그 발송 사실을 회사에 지체없이 알려야 합니다. \uf000 계약자가 청약을 철회한 때에는 회사는 '
 '청약의 철회를 접수한 날부터 3영업일 이내에 납입한 보험료를 돌려드리'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 64},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 108,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
