from langchain_core.documents import Document

chunk = Document(
    page_content=('제11조(보험금 받는 방법의 변경)\n'
 '\uf000 계약자(보험금 지급사유 발생 후에는 보험수익자)는 회 사의 사업방법서에서 정한 바에 따라 보험금의 전부 또는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 55},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 82,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
