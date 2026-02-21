from langchain_core.documents import Document

chunk = Document(
    page_content=('한 계약의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여\n'
 '계약을 해지할 수 있습니다.# 제21조(준용규정)이 특별약관에서 정하지 않은 사항은 보통약관을 따릅니다.- 30 -# 반려견 '
 '슬관절·고관절 치료비 보장 특별약관제1조(보상하는 손해)- ① 회사는 보통약관 제5조(보험금을 지급하지 않은 사유) 제2항 제3호에도 '
 '불구하고\n'
 '- 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전(대퇴 골두 허혈성 괴사 포함)\n'
 '- 또는 기타 이들과 유사한 질병 또는 상해를 원인으로 하여 그 치료를 직접적인'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
