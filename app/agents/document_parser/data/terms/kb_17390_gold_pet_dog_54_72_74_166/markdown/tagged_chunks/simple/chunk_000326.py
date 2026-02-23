from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '| \uf000 제1항의 수술에서 아래에 정한 1. 흡인(吸引) 2. 천자(穿刺) 등의 3. 신경(神經) 차단(NERVE 4. 미용성형 '
 '목적의 5. 피임(避姙) 목적의 수술 6. 검사 및 진단을 위한 7. 제1항 내지 제2항에 해당하지 | 조치 | 않는 | 사항은 '
 '제외합니다. BLOCK) 수술 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등) 시술(체외 충격파 쇄석술 및 변연절제를 |\n'
 '| --- | --- | --- | --- |\n'
 '- \n'
 '동반하지 않은 단순 창상봉합술 등)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000326',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
