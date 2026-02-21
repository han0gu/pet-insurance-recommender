from langchain_core.documents import Document

chunk = Document(
    page_content=('기환급금의 지급)은 제외합니다.- 120 -※ 약관에서 인용된 법·규정은「별표 및 참고」의 「약관에서 인용된 법·규정」에서\n'
 '확인할 수 있습니다.제도성 특별약관- 122 -5-1. [갱신형] 특별약관의 자동갱신 특별약관# 제 1조 (적용대상)이 특별약관은 손해의 '
 '보상을 내용으로 하는 이 계약의 다른 특별약관 중 [갱신형] 특별\n'
 '약관(이하 「갱신형 계약」이라 합니다.)에 대하여 적용합니다.# 제 2조 (자동갱신에 관한 사항)① 이 특별약관은 다음 각 호의 조건을 '
 '충족하고 계약자가 갱신하기 직전 종전의 갱신형'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
