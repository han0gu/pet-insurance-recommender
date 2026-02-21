from langchain_core.documents import Document

chunk = Document(
    page_content=("능력 또는 의사 결정 능력이 없는 상태를<br>말합니다.</p><br><p id='42' data-category='list' "
 "style='font-size:16px'>② 보험수익자가 고의로 피보험자를 해친 경우"),
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
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
