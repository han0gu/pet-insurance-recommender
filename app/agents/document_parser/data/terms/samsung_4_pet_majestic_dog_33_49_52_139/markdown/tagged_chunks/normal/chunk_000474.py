from langchain_core.documents import Document

chunk = Document(
    page_content=('- 데노바이러스 2형 감염증, 코로나바이러스 감염증, 렙토스피라 감염증, 심상사상충 감염\n'
 '- 증, 광견병, 켄넬코프\n'
 '- 14. 기관협착, 누루관시술과 관련된 상해 또는 질병 치료에 대한 비용\n'
 '- 15. 아래의 유전적 또는 발달이상을 원인으로 하는 경우는 보상하지 않습니다.\n'
 '# 가. 뼈와 관절의 영역Wobbler증후군, 팔꿈치 관절형성부전, 팔꿈치 관절 척골 이탈, 팔꿈치 관절요\n'
 '골 이탈, 앞발 허리골의 만곡증, 대퇴골두 괴사증(Legg-calv-perthes\n'
 'disease)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000474',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
