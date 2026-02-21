from langchain_core.documents import Document

chunk = Document(
    page_content=('- 으로 하여 생긴 손해는 보상하지 않습니다.\n'
 '- ② 회사는 다음 중 어느 한 가지 목적의 치료를 위한 상해 입원 수술비 또는 상해 통원\n'
 '- 수술비에 대하여는 보상하지 않습니다.\n'
 '- 1. 건강검진, 예방접종, 인공유산\n'
 '- 2. 영양제, 비타민제, 호르몬 투여, 보신용 투약, 친자 확인을 위한 진단, 불임검사, 불\n'
 '- 임수술, 불임복원술, 보조생식술(체내, 체외 인공수정을 포함합니다), 성장촉진과\n'
 '- 관련된 수술\n'
 '- 3. 아래에 열거된 국민건강보험 비급여 대상으로 신체의 필수 기능개선 목적이 아닌'),
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
 'indexing': {'chunk_id': 'chunk_000332',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
