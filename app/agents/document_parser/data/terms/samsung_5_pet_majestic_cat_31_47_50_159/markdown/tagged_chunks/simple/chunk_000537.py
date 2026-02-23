from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 표시하여야 합니다.\n'
 '- ⑤ 제3항 및 제4항에도 불구하고, 회사가 계약자의 재가입 의사를 확인하지 못한 경우(계\n'
 '- 약자와의 연락두절로 회사의 안내가 계약자에게 도달하지 못한 경우 포함)에는 직전\n'
 '- 계약과 동일한 조건으로 보험계약을 연장합니다. 다만, 보험계약이 연장된 경우 연장\n'
 '- 된 날 기준으로 매년 현재의 예정기초율(적용이율, 적용위험률, 부가보험요율) 적용\n'
 '- 및 반려동물의 연령 증가 등의 사유로 보험요율이 변동될 수 있으며 이 때의 보험료'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000537',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
