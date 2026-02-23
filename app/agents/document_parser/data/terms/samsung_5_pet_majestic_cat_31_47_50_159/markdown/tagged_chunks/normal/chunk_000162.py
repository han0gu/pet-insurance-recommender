from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>[습관성 유산, 불임 및 인공수정 관련 합병증]\n'
 '한국표준질병∙사인분류상의 N96~N98에 해당하는 질병을 말합니다.5. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동② 회사는 다른 '
 '약정이 없으면 피보험자가 직업, 직무 또는 동호회 활동목적으로 아래에열거된 행위로 인하여 각 특별약관별 보험금의 지급사유의 보험금 '
 '지급사유가 발생한\n'
 '때에는 해당 보험금을 지급하지 않습니다.- 1. 전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙벽을 오르내리거나 특수한 기'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
