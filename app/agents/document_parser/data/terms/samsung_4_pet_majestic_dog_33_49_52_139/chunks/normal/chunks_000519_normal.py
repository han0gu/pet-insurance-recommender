from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 제1항 내지 제4항의 창상봉합술은 진료비세부내역서상 보건복지부에서 고시하는 「건 강보험 행위 급여·비급여 목록 및 급여 상대가치점수」 '
 '에서 정한 수가코드 기준을 따 르며, 이 특별약관 체결 시점 이후 보건복지부에서 고시하는 「건강보험 행위 급여·비 급여 목록 및 급여 '
 '상대가치점수」 개정에 따라 수가코드가 변경된 경우에는 개정된 기준을 적용합니다. 다만, 「건강보험 행위 급여·비급여 목록 및 급여 '
 '상대가치점수」 가 폐지되어 보험금 지급사유에 대한 판정이 불가능한 경우 폐지 직전의 관련 법규에 서 정한 분류번호 및 코드를 따릅니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000519',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
