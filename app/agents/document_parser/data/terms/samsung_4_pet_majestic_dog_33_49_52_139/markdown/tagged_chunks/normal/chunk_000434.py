from langchain_core.documents import Document

chunk = Document(
    page_content=('100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 전속하\n'
 '는 전문의를 둔 병원을 말합니다.# 제3조 (창상봉합술의 정의와 장소)- ① 이 특별약관에서 「창상봉합술(3/5cm미만,급여)」 이라 '
 '함은 병원 또는 의원의 의사 또\n'
 '- 는 치과의사의 면허를 가진 자(이하 「의사」 라 하며, 한의사는 제외합니다)에 의하여\n'
 '- 치료가 필요하다고 인정된 경우로서 자택 등에서의 치료가 곤란하여 의료법 제3조(의\n'
 '- 료기관)에서 규정한 국내의 병원 또는 의원에서 의사의 관리 하에 [별표-상해및질병'),
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
 'indexing': {'chunk_id': 'chunk_000434',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
