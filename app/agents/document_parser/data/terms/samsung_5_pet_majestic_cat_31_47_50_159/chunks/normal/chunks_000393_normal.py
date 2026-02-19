from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 건강검진, 예방접종, 인공유산 2. 영양제, 비타민제, 호르몬 투여, 보신용 투약, 친자 확인을 위한 진단, 불임검사, 불 임수술, '
 '불임복원술, 보조생식술(체내, 체외 인공수정을 포함합니다), 성장촉진과 관련된 수술 3. 아래에 열거된 국민건강보험 비급여 대상으로 '
 '신체의 필수 기능개선 목적이 아닌\n'
 '외모개선 목적의 치료를 위한 수술'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000393',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
