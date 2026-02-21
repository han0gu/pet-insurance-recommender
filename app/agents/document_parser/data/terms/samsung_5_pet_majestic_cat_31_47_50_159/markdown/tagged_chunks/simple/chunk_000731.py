from langchain_core.documents import Document

chunk = Document(
    page_content=('⑬ 신경계 · 정신행동의 13개 부위를 말하며, 이를 각각 동일한 신체부위라 한다. 다\n'
 '만, 좌 우의 눈, 귀, 팔, 다리, 손가락, 발가락은 각각 다른 신체부위로 본다.# 3. 기타- 가. 하나의 장해가 관찰 방법에 따라서 '
 '장해분류표상 2가지 이상의 신체부위에서 장\n'
 '- 해로 평가되는 경우에는 그 중 높은 지급률을 적용한다.\n'
 '- 나. 동일한 신체부위에 2가지 이상의 장해가 발생한 경우에는 합산하지 않고 그 중 높\n'
 '- 은 지급률을 적용함을 원칙으로 한다. 그러나 각 신체부위별 판정기준에서 별도'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_000731',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
