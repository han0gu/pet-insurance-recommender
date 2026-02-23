from langchain_core.documents import Document

chunk = Document(
    page_content=('| 용 어 풀 | 이 심신상실 |\n'
 '| 정신병, 정신박약, 심한 등의 사물 는 의사 결정 능력이 없는 상태 | 의식장애 심신장애로 인하여 변별 능력 또 |\n'
 '# \uf000 회사는 다른 약정이 없으면 피보험자가 직업, 직무 또는 동호회 활동목적으로 아래에 열거된 행위로 인하여 제3조(보험금의 '
 '지급사유)의 상해 관련 보험금 지급사\n'
 '유가 발생한 때에는 해당 보험금을 지급하지 않습니다.\n'
 '1. 전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙벽을 오르내리거나 특수- 한 기술, 경험, 사전훈련을 필요로 하는 등반을 '
 '말합니다), 글라이더 조종, 스'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
