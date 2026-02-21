from langchain_core.documents import Document

chunk = Document(
    page_content=("계약자적립액 등의 차이로 인하여 발생한 정산금액(이</p><footer id='33' "
 "style='font-size:14px'>59</footer><p id='34' data-category='paragraph' "
 "style='font-size:16px'>하「정산금액」이라 합니다)을 환급하여 드립니다"),
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
 'indexing': {'chunk_id': 'chunk_000092',
              'chunk_char_len': 167,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
