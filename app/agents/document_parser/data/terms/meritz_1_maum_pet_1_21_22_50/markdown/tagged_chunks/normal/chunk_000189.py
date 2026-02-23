from langchain_core.documents import Document

chunk = Document(
    page_content=('- 료가 정산되기 이전일지라도 새로이 증가 또는 교체된 보험의 목적에 대해 생긴 손해\n'
 '- 를 보상합니다.\n'
 '# 제2조(보험의 목적의 명부)계약자는 항상 보험의 목적의 명부를 비치하여 회사가 열람을 요구할 경우에는 이에 따라\n'
 '야 합니다.# 제3조(보험료의 정산방법)보험료는 보험의 목적의 정보의 변경을 기초로 하여 다음과 같이 정산합니다.- 1. 계약자는 매월 '
 '10일까지 전월말까지의 보험의 목적의 정보의 변경에 관한 서류를 회사\n'
 '- 에 제출하여야 합니다. 그러나 계약이 효력상실 또는 해지된 경우에는 효력상실 또는'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
