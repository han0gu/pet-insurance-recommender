from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다<br>만, 계약자 및 보험료납부자가 법인인 보험계약의 경우<br>에는 보호되지 않습니다.</p><footer id='67' "
 "style='font-size:14px'>82</footer><p id='68' data-category='paragraph' "
 "style='font-size:18px'>무배당 펫퍼민트 Cat&Family보험<br>다이렉트2601 특별약관</p><footer "
 "id='69' style='font-size:14px'>83</footer><footer id='70'"),
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
 'indexing': {'chunk_id': 'chunk_000264',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
