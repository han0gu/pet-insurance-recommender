from langchain_core.documents import Document

chunk = Document(
    page_content=('- 12. 대한민국 이외의 지역에서 발생한 사고 및 손해\n'
 '- 13. 회사는 아래의 치료비, 비용 또는 손해는 보상하지 아니합니다.\n'
 '- 가. 반려동물의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있\n'
 '- 는 증상을 포함합니다. 다만 보험기간 중 최초로 발견된 경우에는 당해 보험기간에 한하여\n'
 '- 보상합니다.)\n'
 '- 나. 질병의 발생일로부터 과거 1년 이내에 예방접종 또는 예방처치를 하지 않아 발생한 아래의\n'
 '- 질병\n'
 '고양이범백혈구감소증, 고양이칼리시바이러스감염증, 고양이바이러스성비기관지염, 고양이백혈병'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
