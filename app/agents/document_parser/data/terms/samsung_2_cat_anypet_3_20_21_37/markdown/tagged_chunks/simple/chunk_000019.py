from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 치석제거 및 치아부정교합 등 치과 치료비용, 구강내 질환\n'
 '【배꼽허니아】 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상\n'
 '【고양이 범백혈구 감소증】 고양이 범백혈구감소증바이러스(FPV) 감염에 의해 발생하는 질환\n'
 '【고양이칼리시 바이러스감염증】 고양이 칼리시바이러스 감염에 의하여 발생하는 질환\n'
 '【고양이바이러스성 비기관지염】 고양이 허피스바이러스 1형 감염에 의한 호흡기 질환\n'
 '【고양이백혈병 바이러스감염증】 고양이 백혈병바이러스에 감염에 의한 조혈기 질환'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
