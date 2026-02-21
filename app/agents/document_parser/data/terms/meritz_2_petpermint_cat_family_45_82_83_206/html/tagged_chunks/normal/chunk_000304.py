from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나<br>계약자, 피보험자 또는 보험수익자의 책임있는 사유로 지급<br>이 지연된 때에는 그 해당기간에 대한 이자는 더하여 '
 '지급<br>하지 않습니다'),
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
 'indexing': {'chunk_id': 'chunk_000304',
              'chunk_char_len': 86,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
