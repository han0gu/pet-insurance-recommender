from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의사를 표시하여야 합니다.\n'
 '- \uf000 제3항 및 제4항에도 불구하고, 회사가 계약자의 재가입 의사를 확인하지 못한 경\n'
 '- 우(계약자와의 연락두절로 회사의 안내가 계약자에게 도달하지 못한 경우 포함)\n'
 '- 에는 직전계약과 동일한 조건으로 보험계약을 연장합니다. 다만, 보험계약이 연\n'
 '- 장된 경우 연장된 날 기준으로 매년 현재의 예정기초율(적용이율, 적용위험률,\n'
 '- 부가보험요율) 적용 및 반려동물의 연령 증가 등의 사유로 보험요율이 변동될 수\n'
 '- 있으며 이 때의 보험료는 "보험료 및 해약환급금 산출방법서"에 따라 산출합니\n'
 '- 다.'),
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
 'indexing': {'chunk_id': 'chunk_000537',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
