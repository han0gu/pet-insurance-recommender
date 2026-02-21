from langchain_core.documents import Document

chunk = Document(
    page_content=('손상으로 양쪽 코의 후각기능을 완전히 잃은<br>경우를 말하며, 후각감퇴는 장해의 대상으로 하지 않는다.<br>3) 양쪽 코의 후각기능은 '
 '후각인지검사, 후각역치검사<br>등을 통해 6개월 이상 고정된 후각의 완전손실이 확<br>인되어야 한다.<br>4) 코의 추상(추한 '
 "모습)장해를 수반한 때에는 기능장해의<br>지급률과 추상(추한 모습)장해의 지급률을 합산한다.</p><h1 id='35' "
 "style='font-size:20px'>4"),
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
 'indexing': {'chunk_id': 'chunk_000947',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
