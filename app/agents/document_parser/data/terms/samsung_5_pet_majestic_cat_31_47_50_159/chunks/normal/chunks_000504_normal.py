from langchain_core.documents import Document

chunk = Document(
    page_content=('보장구분 | 보험금 지급사유 | 지급금액\n'
 '창상봉합술 (3/5cm 미만,급여)(A) | 상해 및질병으로 제3조(창상봉합술의 정의와 장소)에서 정한 '
 '「창상봉합술(3/5cm미만,급여)」 을 받는 경우 | 이 특별약관 가입금액의 10%\n'
 '창상봉합술(급여)(B) | 상해 및 질병으로 제3조(창상봉합술의 정의와 장소)에서 정한 「창상봉합술(급여)」 을 받는 경우 | 이 '
 '특별약관 가입금액의 100%'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000504',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
