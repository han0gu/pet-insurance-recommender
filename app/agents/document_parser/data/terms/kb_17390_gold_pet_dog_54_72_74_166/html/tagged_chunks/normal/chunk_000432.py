from langchain_core.documents import Document

chunk = Document(
    page_content=(". 흡인(吸引)</h1><br><p id='114' data-category='list' style='font-size:14px'>2. "
 '천자(穿刺) 등의 조치<br>3. 신경(神經) 차단(NERVE BLOCK)<br>4. 미용성형 목적의 수술<br>5. 피임(避姙) 목적의 '
 '수술<br>6. 검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)<br>7'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000432',
              'chunk_char_len': 205,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
