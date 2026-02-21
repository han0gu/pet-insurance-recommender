from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우<br>안구가 적출되어 눈자위의 조직요몰(凹沒) 등으로<br>의안마저 끼워 넣을 수 없는 상태이면 “뚜렷한 추<br>상(추한 '
 '모습)”으로, 의안을 끼워 넣을 수 있는 상<br>태이면 “약간의 추상(추한 모습)”으로 지급률을<br>가산한다.<br>12) “눈꺼풀에 '
 '뚜렷한 결손을 남긴 때”에 해당하는 경<br>우에는 추상(추한 모습)장해를 포함하여 장해를 평가<br>한 것으로 보고 추상(추한 '
 '모습)장해를 가산하지 않<br>는다'),
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
 'indexing': {'chunk_id': 'chunk_000929',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
