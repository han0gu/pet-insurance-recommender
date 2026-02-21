from langchain_core.documents import Document

chunk = Document(
    page_content=('- 만, 상해를 직접적인 원인으로 치료를 받은 경우에는 보험계약일을 보장개시일(책임개\n'
 '- 시일)로 합니다. 이 경우 보험계약일은 이 특별약관의 제1회 보험료를 받은 날로 합니\n'
 '- 다.\n'
 '<예시안내>「반려묘 의료비(치과및 구강질환포함)(재가입형)」 에 대한 보장개시일(책임개시일) 계산]# '
 '보험계약일보장개시일(책임개시일)30일# 2022년 8월 1일2022년 8월 31일주) 상해를 직접적인 원인으로 치료를 받은 경우에는 '
 '보장개시일(책임개시일)은 보험계약일로 합니'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000456',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
