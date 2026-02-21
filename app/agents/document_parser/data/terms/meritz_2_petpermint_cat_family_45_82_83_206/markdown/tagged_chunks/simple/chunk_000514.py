from langchain_core.documents import Document

chunk = Document(
    page_content=('이상의 신체부위에서 장해로 평가되는 경우에는 그 중\n'
 '높은 지급률을 적용한다.176- 2) 동일한 신체부위에 2가지 이상의 장해가 발생한 경우에\n'
 '- 는 합산하지 않고 그 중 높은 지급률을 적용함을 원칙\n'
 '- 으로 한다. 그러나 각 신체부위별 판정기준에서 별도\n'
 '- 로 정한 경우에는 그 기준에 따른다.\n'
 '- 3) 하나의 장해가 다른 장해와 통상 파생하는 관계에 있\n'
 '- 는 경우에는 그중 높은 지급률만을 적용하며, 하나의\n'
 '- 장해로 둘 이상의 파생장해가 발생하는 경우 각 파생\n'
 '- 장해의 지급률을 합산한 지급률과 최초 장해의 지급률'),
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
 'indexing': {'chunk_id': 'chunk_000514',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
