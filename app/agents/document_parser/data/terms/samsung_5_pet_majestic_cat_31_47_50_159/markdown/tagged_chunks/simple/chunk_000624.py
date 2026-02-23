from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다.\n'
 '<예시안내># 「반려묘 의료비 확대보장(VRICT)(연간1회한)(재가입형)」 에 대한 보장개시일(책임개시일) '
 '계산]![image](/image/placeholder)\n'
 '보험계약일 보장개시일(책임개시일)\n'
 '30일\n'
 '2022년 8월 1일 2022년 8월 31일# 주) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 '
 '합니\n'
 '다.④ 회사가 지급할 제1항에서 정한 보험금은 피보험자가 부담한 MRI 또는 CT 촬영 당일\n'
 '발생한 의료비에서 4-1. 반려묘의료비(치과및구강질환포함)(재가입형) 특별약관 및'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000624',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
