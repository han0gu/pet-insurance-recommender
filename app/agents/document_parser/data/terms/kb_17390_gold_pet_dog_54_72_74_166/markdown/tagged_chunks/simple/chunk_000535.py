from langchain_core.documents import Document

chunk = Document(
    page_content=('손해보험에 한하며 이하 "반려동물보험 상품"이라 합니다)으로 가입을 할 수 있\n'
 '으며, 회사는 이를 거절할 수 없습니다. 다만, 재가입 계약이 직전계약보다 보장- 내용 및 범위 등이 확대된 경우 확대된 내용에 대해 '
 '회사는 재가입 시점의 인수기\n'
 '- 준에 따라 승낙하거나 일부 보장을 제한할 수 있습니다.\n'
 '- \uf000 회사는 계약자에게 재가입주기(보장내용 변경주기)가 끝나는 날 이전까지 2회 이\n'
 '- 상 재가입 요건, 보장내용 변경내역, 보험료 수준, 재가입 절차 및 재가입 의사'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000535',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
