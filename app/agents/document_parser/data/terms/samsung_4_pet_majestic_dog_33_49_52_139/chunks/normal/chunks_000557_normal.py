from langchain_core.documents import Document

chunk = Document(
    page_content=('. 아래의 질병으로 인하여 발생한 손해는 보상하지 않습니다. (다만, 질병의 발생일 로부터 과거 1년 이내의 예방접종 기록이 있는 '
 '경우에는 보상합니다.) 파보바이러스 감염증, 디스템퍼바이러스 감염증, 파라인플루엔자 감염증, 전염성 간염, 아 데노바이러스 2형 감염증, '
 '코로나바이러스 감염증, 렙토스피라 감염증, 심상사상충 감염 증, 광견병, 켄넬코프'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000557',
              'chunk_char_len': 195,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
