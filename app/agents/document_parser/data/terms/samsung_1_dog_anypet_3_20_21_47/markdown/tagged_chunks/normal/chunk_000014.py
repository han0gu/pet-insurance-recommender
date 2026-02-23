from langchain_core.documents import Document

chunk = Document(
    page_content=('다.)- 10. 대한민국 이외 지역에서 발생한 사고 및 손해\n'
 '- 11. 수의사 자격이 없는 자의 치료행위로 인한 비용 및 그로 인하여 가중된 손해\n'
 '【핵연료물질】 사용된 연료를 포함합니다.\n'
 '【핵연료물질에 의하여 오염된 물질】 원자핵 분열 생성물을 포함합니다.# 회사는 아래의 치료비 및 비용 또는 손해는 보상하지 않습니다.- '
 '1. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약 · 예방 접종비용 및 정기검진, 예방적\n'
 '- 검사를 위한 비용\n'
 '- 2. 임신 · 출산, 제왕절개, 인공유산과 관련된 비용 및 출산 후 증상 치료 비용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
