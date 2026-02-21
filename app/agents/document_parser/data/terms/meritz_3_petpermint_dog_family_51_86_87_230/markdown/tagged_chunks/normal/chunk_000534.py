from langchain_core.documents import Document

chunk = Document(
    page_content=('하여 회사와 계약자간에 합의가 되었을 경우에 적용합니다.\n'
 '\uf000 제1항의 자동갱신 적용대상 특별약관(이하「자동갱신 적\n'
 '용대상 특별약관」이라 합니다)이라 함은 아래의 특별약관\n'
 '을 말합니다.# 【자동갱신 적용대상 특별약관】# ･ 갱신형 펫퍼민트 반려견 배상책임보장 특별약관# 제2조(자동갱신 적용대상 계약의 '
 '자동갱신)\uf000 보장계약이 다음 각 호의 조건을 충족하고, 보장계약이\n'
 '끝나는 날의 전일까지 계약자로부터 별도의 의사표시가 없\n'
 '을 때에는 종전의 자동갱신 적용대상 계약(이하「갱신전 보\n'
 '장계약」이라 합니다)이 끝나는 날의 다음날(이하「갱신'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000534',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
