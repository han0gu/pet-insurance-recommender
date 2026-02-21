from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제2항 및 제3항에서 자동차관리법(하위 법령, 규칙 포함) 및 도로교통법(하위 법\n'
 '- 령, 규칙 포함)변경시 변경된 내용을 적용합니다.\n'
 '- \uf000 피보험자에게 보험사고가 발생했을 경우 그 사고가 이륜자동차를 운전하는 도중에\n'
 '- 발생한 사고인가 아닌가는 계약자 또는 피보험자가 거주하는 관할 경찰서에서 발\n'
 '# 행한 사고처리 확인원등으로 결정합니다.제3조(해지된 특약의 부활(효력회복))\n'
 '회사는 이 특별약관의 부활(효력회복)청약을 받은 경우에는 보험계약의 부활(효력회'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000756',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
