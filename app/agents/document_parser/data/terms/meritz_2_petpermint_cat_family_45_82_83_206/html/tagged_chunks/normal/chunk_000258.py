from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사<br>는 이 계약의 체결, 유지, 보험금 지급 등을 위하여 위 관<br>계 법령에 따라 계약자 및 피보험자의 동의를 받아 '
 '다른 보<br>험회사 및 보험관련단체 등에 개인정보를 제공할 수 있습니다.<br>\uf000 회사는 계약과 관련된 개인정보를 안전하게 '
 "관리하여야<br>합니다.</p><h1 id='58' style='font-size:20px'>제47조(준거법)</h1><br><p "
 "id='59' data-category='paragraph' style='font-size:20px'>이 계약은 대한민국 법에 따라 "
 '규율되고'),
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
 'indexing': {'chunk_id': 'chunk_000258',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
