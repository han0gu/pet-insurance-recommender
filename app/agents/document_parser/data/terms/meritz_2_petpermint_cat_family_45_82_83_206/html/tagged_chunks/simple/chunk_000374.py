from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 재가입을 원하지 않는<br>경우에는 해당 시점으로부터 계약은 해지됩니다(단, 최초연<br>장된 날로부터 90일 '
 "이전에는 계약을 취소 또는 해지할 수<br>있습니다.)</p><br><p id='22' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제8항 내지 제10항에 따라 계약이 해지된 경우 "
 '회사는<br>\uf000<br>보통약관 제35조(해약환급금) 제1항에 따른 해약환급금을<br>계약자에게 지급합니다.</p><p '
 "id='23' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000374',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
