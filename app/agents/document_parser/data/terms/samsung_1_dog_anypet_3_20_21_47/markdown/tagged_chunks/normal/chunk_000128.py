from langchain_core.documents import Document

chunk = Document(
    page_content=('- 액을 보상하여 드립니다.\n'
 '- ② 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경우 동물병원에서 발\n'
 '- 급한 소견서를 제출하여야 합니다.\n'
 '# 제2조(보상하지 않는 손해)회사는 아래의 사유로 인한 손해는 보상하지 않습니다.- 1. 계약자, 피보험자 또는 이들의 가족 또는 '
 '사용인의 고의 또는 중대한 과실\n'
 '- 2. 보험개시일로부터 그 날을 포함하여 30일 이내에 발생한 손해. 단, 이 반려동물 사망위로금 특\n'
 '- 별약관을 갱신하는 경우에는 적용하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000128',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
