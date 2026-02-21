from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사가 건<br>강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계<br>약 또는 전문금융소비자가 체결한 계약은 청약을 '
 "철회할 수<br>없습니다.</p><br><h1 id='74' "
 "style='font-size:20px'>【일반금융소비자】</h1><br><h1 id='75' "
 "style='font-size:20px'>전문금융소비자가 아닌 계약자를 말합니다.</h1><h1 id='76' "
 "style='font-size:20px'>【전문금융소비자】</h1><br><p id='77'"),
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
 'indexing': {'chunk_id': 'chunk_000124',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
