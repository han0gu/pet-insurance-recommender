from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전자문서 및 전자서명의 위조ㆍ변조 여부를 확인할 수<br>있을 것</p><br><h1 id='10' "
 "style='font-size:20px'>【심신상실자 및 심신박약자】</h1><br><p id='11' "
 "data-category='paragraph' style='font-size:20px'>심신상실자(心神喪失者) 또는 "
 '심신박약자(心神薄弱者)라<br>함은 정신병, 정신박약, 심한 의식장애 등의 심신장애로<br>인하여 사물 변별 능력 또는 의사 결정 능력이 '
 "없거나<br>부족한 자를 말합니다.</p><h1 id='12'"),
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
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
