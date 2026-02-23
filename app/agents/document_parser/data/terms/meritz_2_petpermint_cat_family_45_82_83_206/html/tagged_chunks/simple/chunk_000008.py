from langchain_core.documents import Document

chunk = Document(
    page_content=('. 약관에서 정한 대상질병(항목) 분류표의 분류 번호와 다르나 제9차 한국표준질병사인분류의 기준에 따라 분류번호를 동시에 부여가 가능 한 '
 '경우 해당 대상질병(항목) 분류에 포함합 니다. 제10차 개정 이후 약관에서 보상하는 대상질 병(항목) 해당여부는 진단 당시 시행되고 있 '
 '는 한국표준질병사인분류에 따라 판단합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000008',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
