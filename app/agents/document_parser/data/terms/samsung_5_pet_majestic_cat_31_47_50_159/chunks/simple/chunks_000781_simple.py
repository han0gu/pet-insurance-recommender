from langchain_core.documents import Document

chunk = Document(
    page_content=('제 1조 (적용대상)\n'
 '이 특별약관은 손해의 보상을 내용으로 하는 이 계약의 다른 특별약관 중 [갱신형] 특별 약관(이하 「갱신형 계약」이라 합니다.)에 대하여 '
 '적용합니다.\n'
 '제 2조 (자동갱신에 관한 사항)\n'
 '① 이 특별약관은 다음 각 호의 조건을 충족하고 계약자가 갱신하기 직전 종전의 갱신형 특별약관(이하 「갱신전 계약」이라 합니다.)의 '
 '보험기간이 끝나는 날의 전일까지 계 약자로부터 별도의 의사표시가 없을 때에는 갱신전 계약과 동일한 보장내용으로 자동 으로 갱신되는 것으로 '
 '합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 123},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000781',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
