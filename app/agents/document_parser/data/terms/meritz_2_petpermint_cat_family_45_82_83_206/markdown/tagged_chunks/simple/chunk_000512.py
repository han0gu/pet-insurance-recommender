from langchain_core.documents import Document

chunk = Document(
    page_content=('- 인 경우 해당장해 지급률의 20%를 장해지급률로 한다.\n'
 '- 5) 위 4)에 따라 장해지급률이 결정되었으나 그 이후 보\n'
 '- 장받을 수 있는 기간(계약의 효력이 없어진 경우에는\n'
 '- 보험기간이 10년 이상인 계약은 상해 발생일 또는 질\n'
 '- 병의 진단확정일부터 2년 이내로 하고, 보험기간이 10\n'
 '- 년 미만인 계약은 상해 발생일 또는 질병의 진단확정\n'
 '- 일부터 1년 이내)에 장해상태가 더 악화된 때에는 그\n'
 '- 악화된 장해상태를 기준으로 장해지급률을 결정한다.\n'
 '# 2. 신체부위“신체부위”라 함은 ① 눈 ② 귀 ③ 코 ④ 씹어먹거나'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000512',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
