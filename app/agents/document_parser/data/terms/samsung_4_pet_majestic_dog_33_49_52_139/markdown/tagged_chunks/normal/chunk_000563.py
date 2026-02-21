from langchain_core.documents import Document

chunk = Document(
    page_content=('장비용 등 가입동물의 사망 후에 소요된 비용, 각종 증명서류의 작성비용(운송비\n'
 '포함)\n'
 '12. 펫호텔 비용 또는 위탁료, 산책료, 카운슬링 비용, 상담료, 지도료, 문제행동 교정\n'
 '비용 및 이와 동종의 비용\n'
 '13. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. 다만, 질병의 발생일\n'
 '로부터 과거 1년 이내의 예방접종 기록이 있는 경우에는 보상합니다.- \n'
 '파보바이러스 감염증, 디스템퍼바이러스 감염증, 파라인플루엔자 감염증, 전염성 간염, 아\n'
 '데노바이러스 2형 감염증, 코로나바이러스 감염증, 렙토스피라 감염증, 심상사상충 감염'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000563',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
