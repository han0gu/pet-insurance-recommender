from langchain_core.documents import Document

chunk = Document(
    page_content=('있습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하\n'
 '며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니다.<관련법규>[의료법 제3조(의료기관)에 규정한 종합병원]100개 '
 '이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 전속하\n'
 '는 전문의를 둔 병원을 말합니다.# 제3조 (창상봉합술의 정의와 장소)- ① 이 특별약관에서 「창상봉합술(3/5cm미만,급여)」 이라 '
 '함은 병원 또는 의원의 의사 또'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000433',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
