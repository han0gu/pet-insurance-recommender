from langchain_core.documents import Document

chunk = Document(
    page_content=('- 차를 운전(탑승을 포함합니다. 이하 같습니다)하던 중 발생한 급격하고도 우연한\n'
 '- 외래의 상해사고를 직접적인 원인으로 보험계약에서 정한 보험금 지급사유가 발생\n'
 '- 한 경우에 보험금을 지급하지 않습니다. 다만, 피보험자가 이륜자동차를 직업, 직\n'
 '- 무 또는 동호회활동과 출퇴근용도 등 주로 사용하게 된 사실을 회사가 입증하지\n'
 '- 못한 때에는 보험금을 지급합니다.\n'
 '- \uf000 제1항의 이륜자동차라 함은 자동차관리법 시행규칙 제2조에 정한 이륜자동차로 총\n'
 '- 배기량 또는 정격출력의 크기와 관계없이 1인 또는 2인의 사람을 운송하기에 적합'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000751',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
