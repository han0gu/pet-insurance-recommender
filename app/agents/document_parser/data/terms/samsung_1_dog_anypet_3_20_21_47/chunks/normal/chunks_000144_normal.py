from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 않은 경우에는 그가 가입했더라면 의무보험에서 보상했을 금액을 제1항의 '
 '"의무보험에서 보상하는 금액"으로 봅니다.\n'
 '제6조(보험금의 분담)\n'
 '① 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경 우 각 계약에 대하여 다른 계약이 없는 '
 '것으로 하여 각각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다. 이 계약과 다른 계약이 모두 '
 '의무보험인 경 우에도 같습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000144',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
