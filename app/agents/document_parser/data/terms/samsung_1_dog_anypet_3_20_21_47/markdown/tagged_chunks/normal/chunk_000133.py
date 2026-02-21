from langchain_core.documents import Document

chunk = Document(
    page_content=('약관에 따라 보상하지 않습니다.# 제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 31 -당신에게 좋은보험 '
 '삼성화재# 보험료분납 특별약관# 제1조(보험료의 납입)- ① 이 특별약관에 따라 계약자는 보험기간이 1년인 보험 계약에 대하여 보험료를 '
 '제2항에 정한 바에\n'
 '- 따라 나누어 납입할 수 있습니다.\n'
 '- ② 계약자는 이 보험의 보험료 및 해약환급금 산출방법서에서 정한 방법에 의하여 계산된 분납보험료\n'
 '- 를 해당 보험기간 및 분할회수에 따라 아래에 정한 시기까지 납입하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
