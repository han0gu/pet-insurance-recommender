from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 제1회 보험료를 신용카드로 납입한<br>계약의 청약을 철회하는 경우에는 회사는 청약의 철회를 접<br>수한 날부터 '
 '3영업일 이내에 해당 신용카드회사로 하여금<br>대금청구를 하지 않도록 해야 하며, 이 경우 회사는 보험료<br>를 반환한 것으로 '
 "봅니다.</p><br><p id='81' data-category='paragraph' "
 "style='font-size:20px'>\uf000 청약을 철회할 때에 이미 보험금 지급사유가 발생하였으<br>나 계약자가 그 보험금 "
 '지급사유가 발생한 사실을 알지 못<br>한 경우에는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
