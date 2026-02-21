from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 흡인, 천자 등의<br>조치, 신경(神經)차단(NERVE BLOCK), 미용성형 목적의 수<br>술, 피임목적의 수술 및 검사, '
 "진단을 위한 수술(생검, 복<br>강경검사 등)은 제외합니다.</p><br><h1 id='37' "
 "style='font-size:16px'>【용어의 정의】</h1><br><p id='38' data-category='list' "
 "style='font-size:16px'>- 절단(切斷): 특정부위를 잘라 내는 것<br>- 절제(切除): 특정부위를 잘라 없애는 "
 '것<br>- 흡인(吸引): 주사기 등으로'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000654',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
