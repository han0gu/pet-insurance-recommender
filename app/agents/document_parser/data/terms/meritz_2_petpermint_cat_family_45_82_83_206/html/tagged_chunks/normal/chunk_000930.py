from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 안면부의 추상(추한 모습)은 두 가지 장<br>해평가 방법 중 피보험자에게 유리한 것을 적용한다.</p><h1 id='14' "
 "style='font-size:20px'>2. 귀의 장해</h1><h1 id='15' style='font-size:20px'>가"),
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
 'indexing': {'chunk_id': 'chunk_000930',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
