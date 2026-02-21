from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험설계사 등의 행위가 없<br>었다 하더라도 계약자 또는 피보험자가 사실대로 알리<br>지 않거나 부실한 사항을 알렸다고 '
 "인정되는 경우에는<br>계약을 해지할 수 있습니다.</p><br><p id='77' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항에 따라 계약을 해지하였을 때에는 보통약관 제35<br>조(해약환급금) "
 '제1항에 따른 해약환급금을 계약자에게 지<br>급합니다.<br>\uf000 제1항 제1호에 따른 계약의 해지가 보험금 지급사유 '
 '발<br>생 후에 이루어진 경우에'),
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
 'indexing': {'chunk_id': 'chunk_000333',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
