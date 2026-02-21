from langchain_core.documents import Document

chunk = Document(
    page_content=('습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하\n'
 '며, 보험금 지급사유 판정에 드는 비용은 회사가 전액 부담합니다.# <관련법규>[의료법 제3조(의료기관)에 규정한 종합병원]\n'
 '100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다\n'
 '전속하는 전문의를 둔 병원을 말합니다.② 피보험자가 보험기간 중 사망하고, 그 후에 「아나필락시스」 를 직접적인 원인으로 사망한 사실이 '
 '확인된 경우에는 그 사망일을 진단 확정일로 보고 제1조(보험금의 지급'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000313',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
