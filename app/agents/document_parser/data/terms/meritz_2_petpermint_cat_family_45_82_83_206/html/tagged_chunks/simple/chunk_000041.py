from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에<br>서 규정한 국내의 "
 "병원이나 의원 또는 국외의 의료관련법에<br>서 정한 의료기관에서 발급한 것이어야 합니다.</p><h1 id='56' "
 "style='font-size:20px'>제8조(보험금의 지급절차)</h1><br><p id='57' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는 제7조(보험금의 청구)에서 "
 '정한 서류를 접수한<br>때에는 접수증을 드리고 휴대전화 문자메시지'),
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
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
