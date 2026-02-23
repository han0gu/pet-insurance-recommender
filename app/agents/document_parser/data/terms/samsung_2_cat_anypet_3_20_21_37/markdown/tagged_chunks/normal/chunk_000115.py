from langchain_core.documents import Document

chunk = Document(
    page_content=('- 산되기 이전 일지라도 새로이 증가 또는 교체된 피보험자에 대해 생긴 손해를 보상하여 드립니다.\n'
 '# 제2조(피보험자의 명부)계약자는 항상 피보험자 명부를 비치하여 회사가 열람을 요구할 경우에는 이에 따라야 합니다.# 제3조(보험료의 '
 '정산방법)보험료는 피보험자수의 증감을 기초로 하여 다음과 같이 정산합니다.- 1. 계약자는 매월 10일까지 전월말까지의 피보험자수에 관한 '
 '서류를 회사에 제출하여야 합니다. 그\n'
 '- 러나 계약이 효력상실 또는 해지된 경우에는 효력상실 또는 해지일까지의 보험료를 확정하기'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
