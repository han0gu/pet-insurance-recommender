from langchain_core.documents import Document

chunk = Document(
    page_content=('| 창상봉합술(급여)(B) | 상해 및 질병으로 제3조(창상봉합술의 정의와 장소)에서 정한 「창상봉합술(급여)」 을 받는 경우 | 이 '
 '특별약관 가입금액의 100% |\n'
 '| 안면부 창상봉합술 (급여)(C) | 상해 및 질병으로 제3조(창상봉합술의 정의와 장소)에서 정한 「안면부 창상봉합술(급여)」 을 받는 '
 '경우 | 이 특별약관 가입금액의 100% |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000427',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
