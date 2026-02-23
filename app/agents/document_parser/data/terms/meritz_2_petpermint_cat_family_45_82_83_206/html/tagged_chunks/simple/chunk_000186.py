from langchain_core.documents import Document

chunk = Document(
    page_content=('지나기 전까지 회사가 정한 방법에 따라 보험료의<br>자동대출납입을 신청할 수 있으며, 이 경우 제36조(보험계<br>약대출) 제1항에 '
 '따른 보험계약대출금으로 보험료가 자동으<br>로 납입되어 계약은 유효하게 지속됩니다'),
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
 'indexing': {'chunk_id': 'chunk_000186',
              'chunk_char_len': 122,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
