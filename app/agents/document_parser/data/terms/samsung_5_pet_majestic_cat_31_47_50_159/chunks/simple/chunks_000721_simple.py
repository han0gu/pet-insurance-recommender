from langchain_core.documents import Document

chunk = Document(
    page_content=('고양이범백혈구감소증, 고양이칼리시바이러스감염증, 고양이바이러스성비기관지염, 고양 이백혈병바이러스감염증\n'
 '14. 왕진료, 가입동물의 이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및 이 와 동종의 비용 15. 과잉진료행위로 인한 비용 '
 '16. 상병명을 알 수 없는 상해 또는 질병에 대한 치료\n'
 '<용어풀이>\n'
 '배꼽허니아 | 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상\n'
 '고양이 범백혈구감소증 | 고양이 범백혈구감소증바이러스(FPV) 감염에 의해 발생하는 질환'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 115},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000721',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
