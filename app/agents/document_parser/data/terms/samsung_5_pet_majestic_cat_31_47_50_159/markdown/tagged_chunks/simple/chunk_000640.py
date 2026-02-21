from langchain_core.documents import Document

chunk = Document(
    page_content=('니다), 온천요법, 산소요법, 면역요법 등의 대체적 처치에 의한 치료를 위한 비용\n'
 '11. 마이크로칩의 삽입 비용, 안락사를 위한 비용, 장례식비용, 매장비용 등 가입동물\n'
 '의 사망 후에 소요된 비용, 각종 증명서류의 작성비용(운송비 포함)\n'
 '12. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료, 문제행동 교정\n'
 '비용 및 이와 동종의 비용\n'
 '13. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. 다만, 질병의 발생일\n'
 '로부터 과거 1년 이내의 예방접종 기록이 있는 경우에는 보상합니다.-'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000640',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
