from langchain_core.documents import Document

chunk = Document(
    page_content=('- 성과 치료효과를 인정받은 최신 수술기법도 포함됩니다.\n'
 '- \uf000 제1항의 수술에서 아래에 정한 사항은 제외합니다.\n'
 '- 1. 흡인(吸引)\n'
 '- 2. 천자(穿刺) 등의 조치\n'
 '- 3. 신경(神經) 차단(NERVE BLOCK)\n'
 '- 4. 미용성형 목적의 수술\n'
 '- 5. 피임(避姙) 목적의 수술\n'
 '- 6. 검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)\n'
 '- 7. 제1항 내지 제2항에 해당하지 않는 시술(체외 충격파 쇄석술 및 변연절제를\n'
 '- 동반하지 않은 단순 창상봉합술 등)'),
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
 'indexing': {'chunk_id': 'chunk_000339',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
