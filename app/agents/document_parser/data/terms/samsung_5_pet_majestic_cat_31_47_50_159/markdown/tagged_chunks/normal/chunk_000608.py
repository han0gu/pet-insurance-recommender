from langchain_core.documents import Document

chunk = Document(
    page_content=('| 배꼽허니아 | 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상 |\n'
 '| 고양이 범백혈구감소증 | 고양이 범백혈구감소증바이러스(FPV) 감염에 의해 발생하는 질환 |\n'
 '| 고양이 칼리시바이러스감염증 | 고양이 칼리시바이러스 감염에 의하여 발생하는 질환 |\n'
 '| 고양이 바이러스성비기관지염 | 고양이 허피스바이러스 1형 감염에 의한 호흡기 질환 |\n'
 '| 고양이 백혈병바이러스감염증 | 고양이 백혈병바이러스에 감염에 의한 조혈기 질환 |\n'
 '| 잔존유치 | 영구치가 났는데도 불구하고 유치가 남아있어서 발치를 하는 경우 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000608',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
