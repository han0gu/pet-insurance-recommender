from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험설계사 등의 행위가 없<br>었다 하더라도 계약자 또는 피보험자가 사실대로 알리<br>지 않거나 부실한 사항을 알렸다고 '
 "인정되는 경우에는<br>계약을 해지할 수 있습니다.</p><br><p id='52' data-category='list' "
 "style='font-size:20px'>\uf000 제1항에 따라 계약을 해지하였을 때에는 제35조(해약환</p><footer "
 "id='53' style='font-size:14px'>61</footer><p id='54' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
