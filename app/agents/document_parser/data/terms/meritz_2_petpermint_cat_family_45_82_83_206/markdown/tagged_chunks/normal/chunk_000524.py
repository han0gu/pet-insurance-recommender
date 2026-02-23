from langchain_core.documents import Document

chunk = Document(
    page_content=('- 태이면 “약간의 추상(추한 모습)”으로 지급률을\n'
 '- 가산한다.\n'
 '- 12) “눈꺼풀에 뚜렷한 결손을 남긴 때”에 해당하는 경\n'
 '- 우에는 추상(추한 모습)장해를 포함하여 장해를 평가\n'
 '- 한 것으로 보고 추상(추한 모습)장해를 가산하지 않\n'
 '- 는다. 다만, 안면부의 추상(추한 모습)은 두 가지 장\n'
 '- 해평가 방법 중 피보험자에게 유리한 것을 적용한다.\n'
 '# 2. 귀의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 두 귀의 청력을 완전히 잃었을 때 | 80 |'),
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
 'indexing': {'chunk_id': 'chunk_000524',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
