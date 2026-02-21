from langchain_core.documents import Document

chunk = Document(
    page_content=('- 제\n'
 '- 도\n'
 '제3조(보험금을 지급하지 않는 사유)해회사는 보통약관 제1절 일반조항 제5조(보험금을어느 한 가지 목적의 치료를 위한 보험금 지급사유가 '
 '발생한 때에는 보험금을 지급지급하지 않는 사유) 및 다음 중약성KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 79특- 79 -관- '
 '하지 않습니다.\n'
 '- 1. 건강검진, 예방접종, 인공유산\n'
 '- 2. 영양제, 비타민제, 호르몬 투여, 보신용 투약, 친자 확인을 위한 진단, 불임\n'
 '- 검사, 불임수술, 불임복원술, 보조생식술(체내, 체외 인공수정을 포함합니'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000297',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
