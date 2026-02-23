from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제16조(알릴 의무 위반의 효과)\uf000 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 이 계약을\n'
 '해지할 수 있습니다.- 1. 계약자 또는 피보험자가 고의 또는 중대한 과실로 제14조(계약 전 알릴 의무)\n'
 '- 를 위반하고 그 의무가 중요한 사항에 해당하는 경우\n'
 '- 2. 뚜렷한 위험의 증가와 관련된 제15조(상해보험계약 후 알릴 의무) 제1항에서\n'
 '- 정한 계약 후 알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로\n'
 '- 이행하지 않았을때'),
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
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
