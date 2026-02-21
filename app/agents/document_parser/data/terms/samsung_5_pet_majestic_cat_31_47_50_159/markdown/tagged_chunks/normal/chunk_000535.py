from langchain_core.documents import Document

chunk = Document(
    page_content=('- 응한 반려동물보험 상품(보험업감독규정 제1-2조(정의)에서 정한 장기손해보험에 한\n'
 '- 하며 이하 「반려동물보험 상품」 이라 합니다)으로 가입을 할 수 있으며, 회사는 이를\n'
 '- 거절할 수 없습니다. 다만, 재가입 계약이 직전계약보다 보장내용 및 범위 등이 확대\n'
 '- 된 경우 확대된 내용에 대해 회사는 재가입 시점의 인수기준에 따라 승낙하거나 일부\n'
 '- 보장을 제한할 수 있습니다.\n'
 '- ③ 회사는 계약자에게 재가입주기(보장내용 변경주기)가 끝나는 날 이전까지 2회 이상 재'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000535',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
