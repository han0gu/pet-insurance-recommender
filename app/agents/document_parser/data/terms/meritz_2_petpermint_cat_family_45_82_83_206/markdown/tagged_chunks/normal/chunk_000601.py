from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5) 흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때 | 15 |\n'
 '# 나. 장해의 판정기준- 1) “심장 기능을 잃었을 때”라 함은 심장 이식을 한 경\n'
 '- 우를 말한다.\n'
 '- 2) “흉복부장기 또는 비뇨생식기 기능을 잃었을 때” 라\n'
 '- 함은 아래의 경우 중 하나에 해당하는 때를 말한다.\n'
 '- 가) 폐, 신장, 또는 간장의 장기이식을 한 경우\n'
 '- 나) 장기이식을 하지 않고서는 생명유지가 불가능하\n'
 '- 여 혈액투석, 복막투석 등 의료처치를 평생토록\n'
 '- 받아야 할 때'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000601',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
