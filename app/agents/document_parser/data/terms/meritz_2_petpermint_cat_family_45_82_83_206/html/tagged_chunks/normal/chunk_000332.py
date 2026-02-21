from langchain_core.documents import Document

chunk = Document(
    page_content=("것을 방해한 경우, 계약자 또는 피보험자에게</p><footer id='75' "
 "style='font-size:14px'>93</footer><p id='76' data-category='paragraph' "
 "style='font-size:16px'>사실대로 알리지 않게 하였거나 부실한 사항을 알릴<br>것을 권유했을 때"),
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
 'indexing': {'chunk_id': 'chunk_000332',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
