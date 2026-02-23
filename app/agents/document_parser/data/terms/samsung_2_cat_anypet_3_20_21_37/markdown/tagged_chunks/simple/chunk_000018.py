from langchain_core.documents import Document

chunk = Document(
    page_content=('비용- 거. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용\n'
 '- 너. 마이크로칩의 삽입비용, 각종 증빙서류의 작성비용(우송비 포함)\n'
 '- 더. 과잉진료행위로 인한 비용\n'
 '【핵연료물질】 사용된 연료를 포함합니다.\n'
 '【핵연료물질에 의하여 오염된 물질】 원자핵 분열 생성물을 포함합니다.② 회사는 가입동물인 고양이에 대하여 아래의 질병 또는 상해로 인한 '
 '치료비, 비용 또는 손해는 보상\n'
 '하지 아니합니다.- 1. 비뇨기질환(요로결석 등)\n'
 '- 2. 치석제거 및 치아부정교합 등 치과 치료비용, 구강내 질환'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['dental', 'digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
