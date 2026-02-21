from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 피보험자의 치매를 제외한 정신적 기능장해, 선천성 뇌질환 및 심신상실- 2. 성병\n'
 '- 3. 알코올 중독, 습관성 약품 또는 환각제의 복용 및 사용\n'
 '- \uf000 회사는 아래의 의료비로 보험금 지급사유가 발생한 때에는 보험금을 지급하지\n'
 '습니다.\n'
 '1. 질병을 원인으로 하지 않은 신체검사, 예방접종, 인공유산, 불임시술, 제왕절- 개수술\n'
 '- 2. 피로, 권태, 심신허약 등을 치료하기 위한 안정치료비\n'
 '않- \n'
 '# 3. 위생관리, 미모를 위한 성형수술# 4. 정상분만, 치과질환- 별\n'
 '- 제4조(입원의 정의와 장소)\n'
 '- 약'),
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
 'indexing': {'chunk_id': 'chunk_000433',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
