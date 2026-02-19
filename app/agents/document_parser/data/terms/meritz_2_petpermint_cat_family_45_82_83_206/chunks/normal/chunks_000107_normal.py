from langchain_core.documents import Document

chunk = Document(
    page_content=('【부가설명】\n'
 '계약자가 보험수익자가 변경되었음을 회사에 통지하기 전 에 보험금 지급사유가 발생한 경우 회사는 변경 전 보험 수익자에게 보험금을 지급할 '
 '수 있습니다. 회사가 변경 전 보험수익자에게 보험금을 지급한 경우 변경된 보험수 익자에게는 별도로 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 69},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
