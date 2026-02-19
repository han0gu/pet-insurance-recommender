from langchain_core.documents import Document

chunk = Document(
    page_content=('10) 치아에 5개 이상의 결손이 생긴 때 | 5\n'
 '나. 장해의 평가기준\n'
 '1) 씹어먹는 기능의 장해는 윗니(상악치아)와 아랫니(하악치아)의 맞물림(교합), 배 열상태 및 아래턱의 개구운동, 삼킴(연하)운동 등에 '
 '따라 종합적으로 판단하여 결정한다. 2) "씹어먹는 기능에 심한 장해를 남긴 때" 라 함은 심한 개구운동 제한이나 저작 운동 제한으로 '
 '물이나 이에 준하는 음료 이외는 섭취하지 못하는 경우를 말한 다. 3) "씹어먹는 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 '
 '중 하나 이상'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 138},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000892',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
