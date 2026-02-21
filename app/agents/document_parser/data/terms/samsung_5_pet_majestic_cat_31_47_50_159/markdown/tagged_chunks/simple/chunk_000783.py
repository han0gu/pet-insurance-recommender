from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이상 남은 상태\n'
 '- 3) "빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골), 어깨뼈(견갑골)에 뚜렷한 기형이 남\n'
 '- 은 때" 라 함은 방사선 검사로 측정한 각(角) 변형이 20° 이상인 경우를 말\n'
 '- 한다.\n'
 '- 4) 갈비뼈(늑골)의 기형은 그 개수와 정도, 부위 등에 관계없이 전체를 일괄하여\n'
 '- 하나의 장해로 취급한다. 다발성늑골 기형의 경우 각각의 각(角) 변형을 합산\n'
 '- 하지 않고 그 중 가장 높은 각(角) 변형을 기준으로 평가한다.\n'
 '![image](/image/placeholder)\n'
 '제1번(1th)\n'
 '제2번 (2th)'),
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
 'indexing': {'chunk_id': 'chunk_000783',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
