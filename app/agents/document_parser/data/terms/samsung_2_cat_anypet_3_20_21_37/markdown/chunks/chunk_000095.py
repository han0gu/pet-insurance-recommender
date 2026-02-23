from langchain_core.documents import Document

chunk = Document(
    page_content=('- 급한 소견서를 제출하여야 합니다.\n'
 '# 제2조(보상하지 않는 손해)회사는 아래의 사유로 인한 손해는 보상하지 않습니다.- 1. 계약자, 피보험자 또는 이들의 가족 또는 '
 '사용인의 고의 또는 중대한 과실\n'
 '- 2. 보험개시일로부터 그 날을 포함하여 30일 이내에 발생한 손해. 단, 이 반려동물 사망위로금 특\n'
 '- 별약관을 갱신하는 경우에는 적용하지 않습니다.\n'
 '- 3. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 실험 및 이와 유사한 목적으로 이용함으로써\n'
 '- 발생한 손해'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
