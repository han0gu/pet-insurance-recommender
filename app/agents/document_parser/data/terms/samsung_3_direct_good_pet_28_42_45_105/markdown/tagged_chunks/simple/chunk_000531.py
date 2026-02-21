from langchain_core.documents import Document

chunk = Document(
    page_content=('특별약관(이하 「갱신전 계약」 이라 합니다.)의 보험기간이 끝나는 날의 전일까지 계\n'
 '약자로부터 별도의 의사표시가 없을 때에는 갱신전 계약과 동일한 보장내용으로 자동\n'
 '으로 갱신되는 것으로 합니다.- 1. 갱신될 갱신형 특별약관(이하 「갱신계약」 이라 합니다.)의 보험기간이 회사가 이\n'
 '- 보험의 사업방법서에서 정한 기간 내일 것\n'
 '- 2. 갱신전 계약의 보험기간이 끝난 날의 다음날(이하 「갱신일」 이라 합니다)에 피보험\n'
 '- 자의 나이 또는 피보험자의 반려견 나이가 이 보험의 사업방법서에서 정한 범위\n'
 '- 내일 것'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000531',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
