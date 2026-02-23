from langchain_core.documents import Document

chunk = Document(
    page_content=('- 드립니다.\n'
 '- 제1항의 피부병 치료비에 대한 회사의 보장은 보험개시일로부터 30일 이내에 발생한 질병으로 인\n'
 '- 한 손해는 보상하여 드리지 않습니다. 단, 이 피부병 보장 특별약관을 갱신하는 경우에는 적용하지\n'
 '- 않습니다.\n'
 '# 제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 23 -당신에게 좋은보험 삼성화재# 슬관절 수술비용보장 '
 '특별약관# 제1조(보상하는 손해)회사는 보통약관 제5조(보상하지 않는 손해) 제2항 제13호에도 불구하고 슬개골탈구, 십자인대파열, 고'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
