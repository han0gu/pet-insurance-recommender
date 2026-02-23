from langchain_core.documents import Document

chunk = Document(
    page_content=('않은 경우 또는 보험계약을 체결한 후 계약 전 알릴 의무 위반의<br>효과 등으로 보장을 제한할 경우에 한하여 보상하지 않는 질병을 '
 '분류<br>한 표입니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000845',
              'chunk_char_len': 86,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
