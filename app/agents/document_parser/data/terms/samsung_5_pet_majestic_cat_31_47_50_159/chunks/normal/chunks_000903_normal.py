from langchain_core.documents import Document

chunk = Document(
    page_content=('3) "추상(추한 모습)을 남긴 때" 라 함은 상처의 흔적, 화상 등으로 피부의 변색, 모발의 결손, 조직(뼈, 피부 등)의 결손 및 '
 '함몰 등으로 성형수술을 하여도 더 이상 추상(추한 모습)이 없어지지 않는 경우를 말한다. 4) 다발성 반흔 발생시 각 판정부위(얼굴, '
 '머리, 목) 내의 다발성 반흔의 길이 또는 면적은 합산하여 평가한다. 단, 길이가 5mm 미만의 반흔은 합산대상에서 제 외한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000903',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
