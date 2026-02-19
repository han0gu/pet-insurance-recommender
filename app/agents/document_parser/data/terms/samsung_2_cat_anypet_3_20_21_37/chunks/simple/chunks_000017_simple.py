from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 반려동물의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있 는 증상을 포함합니다. 다만 보험기간 중 '
 '최초로 발견된 경우에는 당해 보험기간에 한하여 보상합니다.) 나. 질병의 발생일로부터 과거 1년 이내에 예방접종 또는 예방처치를 하지 '
 '않아 발생한 아래의 질병\n'
 '고양이범백혈구감소증, 고양이칼리시바이러스감염증, 고양이바이러스성비기관지염, 고양이백혈병 바이러스감염증'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
