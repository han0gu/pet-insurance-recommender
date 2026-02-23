from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>204</footer><p id='56' data-category='paragraph' "
 "style='font-size:16px'><붙임></p><h1 id='57' style='font-size:20px'>일상생활 "
 "기본동작(ADLs) 제한 장해평가표</h1><table id='58' "
 "style='font-size:16px'><thead><tr><td>유형</td><td>제한정도에 따른 "
 '지급률</td></tr></thead><tbody><tr><td>이동 동작</td><td>- 특별한'),
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
 'indexing': {'chunk_id': 'chunk_001107',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
