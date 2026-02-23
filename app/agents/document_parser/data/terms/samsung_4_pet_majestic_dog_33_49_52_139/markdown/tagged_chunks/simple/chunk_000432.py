from langchain_core.documents import Document

chunk = Document(
    page_content=('기간을 의미합니다.# 제2조 (보험금 지급에 관한 세부규정)① 피보험자가 「국민건강보험법」 또는 「의료급여법」 을 적용받지 못하는 사고로 '
 '인하\n'
 '여 창상봉합술을 받은 경우, 진단서 및 진료비세부내역서 등을 통해 이 특별약관에서- 95 -# 정한 수가코드를 확인할 수 있는 경우 '
 '회사는 제3조(창상봉합술의 정의와 장소)에서\n'
 '정한 치료에 포함하여 보장합니다.<예시안내>자동차보험 또는 산재보험 등 「국민건강보험법」 또는 「의료급여법」 을 적용받지 못하는 사고로'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000432',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
