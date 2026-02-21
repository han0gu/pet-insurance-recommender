from langchain_core.documents import Document

chunk = Document(
    page_content=('. 건강검진, 예방접종, 인공유산<br>2. 영양제, 비타민제, 호르몬 투여, 보신용 투약, 친자 확인을 위한 진단, 불임<br>검사, '
 '불임수술, 불임복원술, 보조생식술(체내, 체외 인공수정을 포함합니<br>다), 성장촉진과 관련된 수술<br>3. 아래에 열거된 '
 '국민건강보험 비급여 대상으로 신체의 필수 기능개선 목적이<br>아닌 외모개선 목적의 치료를 위한 수술<br>가. 쌍꺼풀수술(이중검수술'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000404',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
