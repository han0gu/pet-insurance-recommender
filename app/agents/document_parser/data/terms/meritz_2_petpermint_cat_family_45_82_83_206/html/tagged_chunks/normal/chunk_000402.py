from langchain_core.documents import Document

chunk = Document(
    page_content=("따라 회사는 채권자에게 해약환급금을 지급하<br>게 됩니다.</p><h1 id='56' "
 "style='font-size:20px'>제20조(계약자의 임의해지)</h1><br><p id='57' "
 "data-category='paragraph' style='font-size:16px'>계약자는 계약이 소멸하기 전에는 언제든지 계약을 "
 '해지할<br>수 있으며, 이 경우 회사는 보통약관 제35조(해약환급금)<br>제1항에 의한 해약환급금을 계약자에게 지급합니다'),
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
 'indexing': {'chunk_id': 'chunk_000402',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
