from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[창상봉합술]\n'
 '창상봉합술이란 상처로 인해 벌어지거나 수술을 위해 벤 조직을 꿰매어 맞추어 주는 것을 말합니 다. [안면부] 안면부란 이마를 포함하여 '
 '경부(목)까지의 얼굴 부분을 말합니다.\n'
 '② 제1항의 「연간」 이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의 기간을 의미합니다.\n'
 '제2조 (보험금 지급에 관한 세부규정)\n'
 '① 피보험자가 「국민건강보험법」 또는 「의료급여법」 을 적용받지 못하는 사고로 인하 여 창상봉합술을 받은 경우, 진단서 및 '
 '진료비세부내역서 등을 통해 이 특별약관에서'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 95},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000511',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
