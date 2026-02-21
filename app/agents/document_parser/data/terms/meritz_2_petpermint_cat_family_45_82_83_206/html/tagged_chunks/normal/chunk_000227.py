from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나, 순수보장성보험 등 보험상품의<br>종류에 따라 보험계약대출이 제한될 수도 있습니다.<br>\uf000 계약자는 제1항에 따른 '
 '보험계약대출금과 그 이자를 언<br>제든지 상환할 수 있으며 상환하지 않은 때에는 회사는 보<br>험금, 해약환급금 등의 지급사유가 '
 "발생한 날에 지급금에서</p><footer id='9' style='font-size:14px'>77</footer><p id='10' "
 "data-category='paragraph' style='font-size:20px'>보험계약대출의 원금과 이자를 차감할 수"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000227',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
